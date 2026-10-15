import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import pino from "pino";
import { SherpaOnnxTTS, type SherpaTtsPreset } from "./sherpa-tts.js";

const KOKORO_FILES = ["model.onnx", "voices.bin", "tokens.txt"];
const PIPER_FILES = ["pl_PL-mc_speech-medium.onnx", "tokens.txt"];

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

interface CreateTtsOptions {
  preset?: SherpaTtsPreset;
  files?: string[];
  sampleRate?: number;
}

function createModelDir(files: string[]): string {
  const modelDir = mkdtempSync(join(tmpdir(), "paseo-tts-"));
  directories.push(modelDir);
  for (const name of files) writeFileSync(join(modelDir, name), "");
  mkdirSync(join(modelDir, "espeak-ng-data"));
  return modelDir;
}

function createTts(
  numThreads?: number,
  beforeGenerate?: () => Promise<void>,
  options: CreateTtsOptions = {},
) {
  const { preset = "kokoro-en-v0_19", files = KOKORO_FILES, sampleRate = 24000 } = options;
  const modelDir = createModelDir(files);
  let nativeConfig: unknown;
  const requests: unknown[] = [];
  const released: boolean[] = [];
  const tts = new SherpaOnnxTTS(
    { preset, modelDir, numThreads },
    pino({ level: "silent" }),
    () => ({
      OfflineTts: class {
        sampleRate = sampleRate;
        free() {
          released.push(true);
        }
        constructor(config: unknown) {
          nativeConfig = config;
        }
        async generateAsync(request: unknown) {
          await beforeGenerate?.();
          return this.generate(request);
        }
        generate(request: unknown) {
          requests.push(request);
          return { samples: Float32Array.from([0, 0.5, -0.5, 0.25]), sampleRate };
        }
      },
    }),
  );
  return { tts, requests, nativeConfig, released, modelDir };
}

async function yieldNativeQueue(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
}

function holdNativeSynthesis() {
  const pending: Array<() => void> = [];
  function wait() {
    return new Promise<void>((resolve) => pending.push(resolve));
  }
  return { pending, wait };
}

describe("SherpaOnnxTTS", () => {
  it("requests Electron-compatible copied samples and returns PCM audio", async () => {
    const { tts, requests } = createTts();
    const result = await tts.synthesizeSpeech("hello");
    expect(requests).toEqual([{ text: "hello", sid: 0, speed: 1, enableExternalBuffer: false }]);
    expect(result.format).toBe("pcm;rate=24000");
    const chunks: Buffer[] = [];
    for await (const chunk of result.stream) chunks.push(chunk);
    expect(Buffer.concat(chunks).length).toBe(8);
  });

  it("keeps the event loop available and serializes native synthesis", async () => {
    const { pending, wait } = holdNativeSynthesis();
    const { tts, requests } = createTts(undefined, wait);
    const first = tts.synthesizeSpeech("first");
    const second = tts.synthesizeSpeech("second");
    try {
      await yieldNativeQueue();
      expect(requests).toEqual([]);
      expect(pending).toHaveLength(1);
      pending.shift()!();
      const audio = await first;
      audio.stream.destroy();
      await yieldNativeQueue();
      expect(requests).toHaveLength(1);
      expect(pending).toHaveLength(1);
      pending.shift()!();
      (await second).stream.destroy();
      expect(requests.map((request) => (request as { text: string }).text)).toEqual([
        "first",
        "second",
      ]);
    } finally {
      for (const release of pending) release();
      await Promise.allSettled([first, second]);
      tts.free();
    }
  });

  it("continues queued synthesis after a native failure", async () => {
    let attempts = 0;
    const { tts, requests } = createTts(undefined, async () => {
      attempts += 1;
      if (attempts === 1) throw new Error("generation failed");
    });
    const first = tts.synthesizeSpeech("first");
    const second = tts.synthesizeSpeech("second");
    await expect(first).rejects.toThrow("generation failed");
    (await second).stream.destroy();
    expect(requests).toEqual([{ text: "second", sid: 0, speed: 1, enableExternalBuffer: false }]);
    tts.free();
  });

  it("waits for active native synthesis before freeing and drops queued synthesis", async () => {
    const { pending, wait } = holdNativeSynthesis();
    const { tts, requests, released } = createTts(undefined, wait);
    const first = tts.synthesizeSpeech("first");
    const second = tts.synthesizeSpeech("second");
    const rejected = expect(second).rejects.toThrow("TTS provider closed");
    await yieldNativeQueue();
    tts.free();
    expect(released).toEqual([]);
    pending.shift()!();
    (await first).stream.destroy();
    await rejected;
    await yieldNativeQueue();
    expect(requests).toHaveLength(1);
    expect(released).toEqual([true]);
    await expect(tts.synthesizeSpeech("third")).rejects.toThrow("TTS provider closed");
  });

  it("configures the kokoro engine for Kokoro models", () => {
    const { nativeConfig, modelDir } = createTts();
    expect(nativeConfig).toMatchObject({
      model: {
        kokoro: {
          model: join(modelDir, "model.onnx"),
          voices: join(modelDir, "voices.bin"),
          tokens: join(modelDir, "tokens.txt"),
          dataDir: join(modelDir, "espeak-ng-data"),
        },
      },
    });
    expect(nativeConfig).not.toHaveProperty("model.vits");
  });

  it("configures the vits engine for Piper voices and reports the model sample rate", async () => {
    const { tts, nativeConfig, modelDir } = createTts(undefined, undefined, {
      preset: "piper-pl_pl-mc_speech-medium",
      files: PIPER_FILES,
      sampleRate: 22050,
    });
    expect(nativeConfig).toMatchObject({
      model: {
        vits: {
          model: join(modelDir, "pl_PL-mc_speech-medium.onnx"),
          tokens: join(modelDir, "tokens.txt"),
          dataDir: join(modelDir, "espeak-ng-data"),
        },
      },
    });
    expect(nativeConfig).not.toHaveProperty("model.kokoro");
    const result = await tts.synthesizeSpeech("Dzień dobry");
    expect(result.format).toBe("pcm;rate=22050");
    result.stream.destroy();
  });

  it("fails fast when a Piper voice is missing its model file", () => {
    expect(() =>
      createTts(undefined, undefined, {
        preset: "piper-pl_pl-mc_speech-medium",
        files: ["tokens.txt"],
      }),
    ).toThrow("Missing TTS model");
  });

  it.each([undefined, 4])(
    "passes the requested thread count at the native model boundary (%s)",
    (threads) => {
      const { nativeConfig } = createTts(threads);
      expect(nativeConfig).toMatchObject({ model: { numThreads: threads ?? 2, provider: "cpu" } });
      expect(nativeConfig).not.toHaveProperty("numThreads");
      expect(nativeConfig).not.toHaveProperty("provider");
    },
  );
});
