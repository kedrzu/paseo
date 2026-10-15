import type pino from "pino";
import { Readable } from "node:stream";
import { existsSync } from "node:fs";
import path from "node:path";

import type { SpeechStreamResult, TextToSpeechProvider } from "../../../speech-provider.js";
import { chunkBuffer, float32ToPcm16le } from "../../../audio.js";
import { getSherpaOnnxModelSpec, type LocalTtsModelId } from "./model-catalog.js";
import { loadSherpaOnnxNode } from "./sherpa-onnx-node-loader.js";

export type SherpaTtsPreset = LocalTtsModelId;

export interface SherpaTtsConfig {
  preset: SherpaTtsPreset;
  modelDir: string;
  speakerId?: number;
  speed?: number;
  lengthScale?: number;
  numThreads?: number;
}

function isPositiveFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function assertFileExists(filePath: string, label: string): void {
  if (!existsSync(filePath)) {
    throw new Error(`Missing ${label}: ${filePath}`);
  }
}

function buildEngineConfig(config: SherpaTtsConfig): Record<string, unknown> {
  const spec = getSherpaOnnxModelSpec(config.preset);
  if (spec.kind !== "tts") {
    throw new Error(`Local speech model ${config.preset} is not a TTS model`);
  }
  const files = spec.tts;
  const inModelDir = (rel: string) => path.join(config.modelDir, rel);
  const model = inModelDir(files.model);
  const tokens = inModelDir(files.tokens);
  const dataDir = inModelDir(files.dataDir);

  assertFileExists(model, "TTS model");
  assertFileExists(tokens, "TTS tokens");
  assertFileExists(dataDir, "TTS espeak-ng dataDir");
  const lengthScale = config.lengthScale ?? 1.0;

  switch (files.engine) {
    case "kokoro": {
      const voices = inModelDir(files.voices);
      assertFileExists(voices, "TTS voices");
      return { kokoro: { model, voices, tokens, dataDir, lengthScale } };
    }
    case "vits":
      return { vits: { model, tokens, dataDir, lengthScale } };
  }
}

interface SherpaOfflineTtsNative {
  sampleRate?: number;
  generateAsync: (args: {
    text: string;
    sid: number;
    speed: number;
    enableExternalBuffer: boolean;
  }) => Promise<{ samples?: Float32Array | number[]; sampleRate?: number } | undefined>;
  free?: () => void;
}

export class SherpaOnnxTTS implements TextToSpeechProvider {
  private readonly tts: SherpaOfflineTtsNative;
  private readonly speakerId: number;
  private readonly speed: number;
  private readonly logger: pino.Logger;
  private synthesisQueue: Promise<void> = Promise.resolve();
  private closed = false;

  constructor(
    config: SherpaTtsConfig,
    logger: pino.Logger,
    loadNative: () => Pick<
      ReturnType<typeof loadSherpaOnnxNode>,
      "OfflineTts"
    > = loadSherpaOnnxNode,
  ) {
    this.logger = logger.child({ module: "speech", provider: "local", component: "tts" });
    this.speakerId = config.speakerId ?? 0;
    this.speed = config.speed ?? 1.0;

    const sherpa = loadNative();
    if (typeof sherpa.OfflineTts !== "function") {
      throw new Error("sherpa-onnx-node OfflineTts is unavailable");
    }

    const modelConfig = {
      // The native parser reads these under model, despite the upstream JS typedef.
      numThreads: config.numThreads ?? 2,
      provider: "cpu",
      ...buildEngineConfig(config),
    };

    const offlineTtsConfig = {
      model: modelConfig,
      maxNumSentences: 1,
    };

    this.tts = new (
      sherpa as unknown as { OfflineTts: new (config: unknown) => SherpaOfflineTtsNative }
    ).OfflineTts(offlineTtsConfig);
    this.logger.info(
      { preset: config.preset, modelDir: config.modelDir },
      "Sherpa offline TTS initialized",
    );
  }

  async synthesizeSpeech(text: string): Promise<SpeechStreamResult> {
    const trimmed = text.trim();
    if (!trimmed) {
      throw new Error("Cannot synthesize empty text");
    }

    // The native handle is not reentrant. Keep prefetch off the worker event loop
    // while allowing only one generation at a time on this model.
    const result = this.synthesisQueue.then(() => this.generateSpeech(trimmed));
    this.synthesisQueue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private async generateSpeech(text: string): Promise<SpeechStreamResult> {
    if (this.closed) {
      throw new Error("TTS provider closed");
    }
    const audio = await this.tts.generateAsync({
      text,
      sid: this.speakerId,
      speed: this.speed,
      // Electron rejects native external-backed typed arrays. Request a copied buffer
      // from sherpa itself instead of trying to clone after generate() returns.
      enableExternalBuffer: false,
    });
    let rawSamples: Float32Array | null = null;
    if (audio && audio.samples instanceof Float32Array) {
      rawSamples = audio.samples;
    } else if (audio && Array.isArray(audio.samples)) {
      rawSamples = Float32Array.from(audio.samples);
    }
    // Copy to avoid "External buffers are not allowed" when sherpa-onnx
    // returns a Float32Array backed by native memory.
    const samples = rawSamples ? Float32Array.from(rawSamples) : null;
    // Models differ (Kokoro 24 kHz, Piper medium 22.05 kHz), so the rate comes from the
    // model and is never assumed. Clients play PCM at the rate in the format string.
    const sampleRate = [audio?.sampleRate, this.tts.sampleRate].find(isPositiveFiniteNumber);

    if (!samples) {
      throw new Error("Unexpected sherpa TTS output: missing Float32 samples");
    }
    if (sampleRate === undefined) {
      throw new Error("Unexpected sherpa TTS output: missing sample rate");
    }

    const pcm16 = float32ToPcm16le(samples);
    const chunkBytes = Math.max(2, Math.round(sampleRate * 0.05) * 2); // ~50ms
    const chunks = chunkBuffer(pcm16, chunkBytes);

    return {
      stream: Readable.from(chunks),
      format: `pcm;rate=${sampleRate}`,
    };
  }

  free(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    // Let the active native call finish before releasing its handle. Queued
    // calls observe closed and never enter native code.
    void this.synthesisQueue.then(() => {
      try {
        this.tts.free?.();
      } catch {
        // ignore
      }
      return undefined;
    });
  }
}
