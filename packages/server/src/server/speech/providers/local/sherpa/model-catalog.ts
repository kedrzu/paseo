import { z } from "zod";

export type SherpaOnnxModelKind = "stt-offline" | "tts";

type DefaultModelRole = "stt" | "tts";

// Paths are relative to the extracted model directory and map onto the sherpa-onnx
// OfflineTts model config of the same engine name.
type SherpaOnnxTtsEngineFiles =
  | { engine: "kokoro"; model: string; voices: string; tokens: string; dataDir: string }
  | { engine: "vits"; model: string; tokens: string; dataDir: string };

interface SherpaOnnxCatalogEntryBase {
  archiveUrl: string;
  extractedDir: string;
  requiredFiles: string[];
  description: string;
  defaultFor?: DefaultModelRole;
}

type SherpaOnnxCatalogEntry =
  | (SherpaOnnxCatalogEntryBase & { kind: "stt-offline" })
  | (SherpaOnnxCatalogEntryBase & { kind: "tts"; tts: SherpaOnnxTtsEngineFiles });

const PIPER_ARCHIVE_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

// Piper voices ship as sherpa-onnx VITS archives named after the Piper voice
// (`vits-piper-pl_PL-gosia-medium`). Model ids are lowercase because the id schema
// lowercases input.
function piperVoice(voice: string, description: string) {
  const extractedDir = `vits-piper-${voice}`;
  const model = `${voice}.onnx`;
  return {
    kind: "tts",
    archiveUrl: `${PIPER_ARCHIVE_BASE_URL}/${extractedDir}.tar.bz2`,
    extractedDir,
    requiredFiles: [model, "tokens.txt", "espeak-ng-data"],
    description,
    tts: { engine: "vits", model, tokens: "tokens.txt", dataDir: "espeak-ng-data" },
  } as const satisfies SherpaOnnxCatalogEntry;
}

export const SHERPA_ONNX_MODEL_CATALOG = {
  "parakeet-tdt-0.6b-v2-int8": {
    kind: "stt-offline",
    archiveUrl:
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
    extractedDir: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    requiredFiles: ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
    description: "NVIDIA Parakeet TDT v2 (offline NeMo transducer, English).",
    defaultFor: "stt",
  },
  "parakeet-tdt-0.6b-v3-int8": {
    kind: "stt-offline",
    archiveUrl:
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
    extractedDir: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    requiredFiles: ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
    description:
      "NVIDIA Parakeet TDT v3 (offline NeMo transducer, 25 European languages, auto-detected).",
  },
  "kokoro-en-v0_19": {
    kind: "tts",
    archiveUrl:
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
    extractedDir: "kokoro-en-v0_19",
    requiredFiles: ["model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data"],
    description: "Kokoro TTS (higher quality; larger).",
    defaultFor: "tts",
    tts: {
      engine: "kokoro",
      model: "model.onnx",
      voices: "voices.bin",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
    },
  },
  "piper-pl_pl-mc_speech-medium": piperVoice(
    "pl_PL-mc_speech-medium",
    "Piper mc_speech (Polish, single speaker, 22.05 kHz).",
  ),
  "piper-pl_pl-gosia-medium": piperVoice(
    "pl_PL-gosia-medium",
    "Piper gosia (Polish, single speaker, 22.05 kHz).",
  ),
  "piper-pl_pl-darkman-medium": piperVoice(
    "pl_PL-darkman-medium",
    "Piper darkman (Polish, single speaker, 22.05 kHz).",
  ),
} as const satisfies Record<string, SherpaOnnxCatalogEntry>;

export type SherpaOnnxModelId = keyof typeof SHERPA_ONNX_MODEL_CATALOG;
export type LocalSpeechModelId = SherpaOnnxModelId;

type ModelIdByKind<K extends SherpaOnnxModelKind> = {
  [Id in SherpaOnnxModelId]: (typeof SHERPA_ONNX_MODEL_CATALOG)[Id]["kind"] extends K ? Id : never;
}[SherpaOnnxModelId];

export type LocalSttModelId = ModelIdByKind<"stt-offline">;
export type LocalTtsModelId = ModelIdByKind<"tts">;

const ALL_MODEL_IDS: SherpaOnnxModelId[] = Object.keys(SHERPA_ONNX_MODEL_CATALOG).filter(
  (k): k is SherpaOnnxModelId => k in SHERPA_ONNX_MODEL_CATALOG,
);

function isLocalSttModelId(id: SherpaOnnxModelId): id is LocalSttModelId {
  return SHERPA_ONNX_MODEL_CATALOG[id].kind !== "tts";
}

function isLocalTtsModelId(id: SherpaOnnxModelId): id is LocalTtsModelId {
  return SHERPA_ONNX_MODEL_CATALOG[id].kind === "tts";
}

export const LOCAL_STT_MODEL_IDS: LocalSttModelId[] = ALL_MODEL_IDS.filter(isLocalSttModelId);

export const LOCAL_TTS_MODEL_IDS: LocalTtsModelId[] = ALL_MODEL_IDS.filter(isLocalTtsModelId);

function resolveDefaultModelId(role: "stt"): LocalSttModelId;
function resolveDefaultModelId(role: "tts"): LocalTtsModelId;
function resolveDefaultModelId(role: DefaultModelRole): SherpaOnnxModelId {
  const match = ALL_MODEL_IDS.find((id) => {
    const entry: SherpaOnnxCatalogEntry = SHERPA_ONNX_MODEL_CATALOG[id];
    return entry.defaultFor === role;
  });
  if (!match) {
    throw new Error(`No default model configured for role '${role}'`);
  }
  return match;
}

export const DEFAULT_LOCAL_STT_MODEL = resolveDefaultModelId("stt");
export const DEFAULT_LOCAL_TTS_MODEL = resolveDefaultModelId("tts");

function createModelIdSchema<T extends string>(modelIds: readonly T[]): z.ZodType<T, string> {
  const validIds = new Set<string>(modelIds);
  return z
    .string()
    .trim()
    .toLowerCase()
    .refine((value) => validIds.has(value), {
      message: "Invalid model id",
    })
    .transform((value) => value as T);
}

export const LocalSttModelIdSchema = createModelIdSchema(LOCAL_STT_MODEL_IDS);
export const LocalTtsModelIdSchema = createModelIdSchema(LOCAL_TTS_MODEL_IDS);

export type SherpaOnnxModelSpec = SherpaOnnxCatalogEntry & {
  id: SherpaOnnxModelId;
};

export function listSherpaOnnxModels(): SherpaOnnxModelSpec[] {
  return ALL_MODEL_IDS.map((id) => Object.assign({ id }, SHERPA_ONNX_MODEL_CATALOG[id]));
}

export function getSherpaOnnxModelSpec(id: SherpaOnnxModelId): SherpaOnnxModelSpec {
  const spec = SHERPA_ONNX_MODEL_CATALOG[id];
  if (!spec) {
    throw new Error(`Unknown local speech model id: ${id}`);
  }
  return {
    id,
    ...spec,
  };
}
