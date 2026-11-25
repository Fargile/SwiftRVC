// c bridge around libbeatrice.a (20a2 / 20b1 / 20rc0)

#include "lib/beatrice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kBinsPerSemitone =
    static_cast<double>(BEATRICE_PITCH_BINS_PER_OCTAVE) / 12.0;

static inline double hz_to_beatrice_bin(double hz) {
  if (hz <= 10.0) return 0.0;
  double midi = 69.0 + 12.0 * std::log2(hz / 440.0);
  return (midi - 33.0) * kBinsPerSemitone;
}
	
  
		
  
struct BeatriceBridge {
  int version = -1;  // 0=20a2, 1=20b1, 2=20rc0
  int n_speakers = 0;
  int current_speaker = 0;
  bool loaded = false;
  
  double pitch_shift = 0.0;
  double formant_shift = 0.0;
  double intonation = 1.0;
  int vq_neighbors = 2;
  double avg_src_pitch_midi = 50.0;
  double min_src_pitch_midi = 36.0;
  double max_src_pitch_midi = 84.0;
  double pitch_floor_hz = 0.0;

  // 20a2
  Beatrice20a2_PhoneExtractor* a2_phone = nullptr;
  Beatrice20a2_PhoneContext1* a2_phone_ctx = nullptr;
  Beatrice20a2_PitchEstimator* a2_pitch = nullptr;
  Beatrice20a2_PitchContext1* a2_pitch_ctx = nullptr;
  Beatrice20a2_WaveformGenerator* a2_wf = nullptr;
  Beatrice20a2_WaveformContext1* a2_wf_ctx = nullptr;

  // 20b1
  Beatrice20b1_PhoneExtractor* b1_phone = nullptr;
  Beatrice20b1_PhoneContext1* b1_phone_ctx = nullptr;
  Beatrice20b1_PitchEstimator* b1_pitch = nullptr;
  Beatrice20b1_PitchContext1* b1_pitch_ctx = nullptr;
  Beatrice20b1_WaveformGenerator* b1_wf = nullptr;
  Beatrice20b1_WaveformContext1* b1_wf_ctx = nullptr;

  // 20rc0
  Beatrice20rc0_PhoneExtractor* rc0_phone = nullptr;
  Beatrice20rc0_PhoneContext1* rc0_phone_ctx = nullptr;
  Beatrice20rc0_PitchEstimator* rc0_pitch = nullptr;
  Beatrice20rc0_PitchContext1* rc0_pitch_ctx = nullptr;
  Beatrice20rc0_WaveformGenerator* rc0_wf = nullptr;
  Beatrice20rc0_WaveformContext1* rc0_wf_ctx = nullptr;
  Beatrice20rc0_EmbeddingSetter* rc0_setter = nullptr;
  Beatrice20rc0_EmbeddingContext* rc0_emb_ctx = nullptr;

  std::vector<float> spk_emb;
  std::vector<float> codebooks;
  std::vector<float> formant_emb;
  std::vector<float> kv_emb;

  void unload() {
    loaded = false;
    version = -1;
    n_speakers = 0;

    if (a2_phone_ctx) { Beatrice20a2_DestroyPhoneContext1(a2_phone_ctx); a2_phone_ctx = nullptr; }
    if (a2_phone) { Beatrice20a2_DestroyPhoneExtractor(a2_phone); a2_phone = nullptr; }
    if (a2_pitch_ctx) { Beatrice20a2_DestroyPitchContext1(a2_pitch_ctx); a2_pitch_ctx = nullptr; }
    if (a2_pitch) { Beatrice20a2_DestroyPitchEstimator(a2_pitch); a2_pitch = nullptr; }
    if (a2_wf_ctx) { Beatrice20a2_DestroyWaveformContext1(a2_wf_ctx); a2_wf_ctx = nullptr; }
    if (a2_wf) { Beatrice20a2_DestroyWaveformGenerator(a2_wf); a2_wf = nullptr; }

    if (b1_phone_ctx) { Beatrice20b1_DestroyPhoneContext1(b1_phone_ctx); b1_phone_ctx = nullptr; }
    if (b1_phone) { Beatrice20b1_DestroyPhoneExtractor(b1_phone); b1_phone = nullptr; }
    if (b1_pitch_ctx) { Beatrice20b1_DestroyPitchContext1(b1_pitch_ctx); b1_pitch_ctx = nullptr; }
    if (b1_pitch) { Beatrice20b1_DestroyPitchEstimator(b1_pitch); b1_pitch = nullptr; }
    if (b1_wf_ctx) { Beatrice20b1_DestroyWaveformContext1(b1_wf_ctx); b1_wf_ctx = nullptr; }
    if (b1_wf) { Beatrice20b1_DestroyWaveformGenerator(b1_wf); b1_wf = nullptr; }

    if (rc0_phone_ctx) { Beatrice20rc0_DestroyPhoneContext1(rc0_phone_ctx); rc0_phone_ctx = nullptr; }
    if (rc0_phone) { Beatrice20rc0_DestroyPhoneExtractor(rc0_phone); rc0_phone = nullptr; }
    if (rc0_pitch_ctx) { Beatrice20rc0_DestroyPitchContext1(rc0_pitch_ctx); rc0_pitch_ctx = nullptr; }
    if (rc0_pitch) { Beatrice20rc0_DestroyPitchEstimator(rc0_pitch); rc0_pitch = nullptr; }
    if (rc0_wf_ctx) { Beatrice20rc0_DestroyWaveformContext1(rc0_wf_ctx); rc0_wf_ctx = nullptr; }
    if (rc0_wf) { Beatrice20rc0_DestroyWaveformGenerator(rc0_wf); rc0_wf = nullptr; }
    if (rc0_emb_ctx) { Beatrice20rc0_DestroyEmbeddingContext(rc0_emb_ctx); rc0_emb_ctx = nullptr; }
    if (rc0_setter) { Beatrice20rc0_DestroyEmbeddingSetter(rc0_setter); rc0_setter = nullptr; }

    spk_emb.clear(); spk_emb.shrink_to_fit();
    codebooks.clear(); codebooks.shrink_to_fit();
    formant_emb.clear(); formant_emb.shrink_to_fit();
    kv_emb.clear(); kv_emb.shrink_to_fit();
  }

  void apply_rc0_speaker(int spk_id) {
    if (!loaded || version != 2 || n_speakers <= 0) return;
    spk_id = std::clamp(spk_id, 0, n_speakers - 1);
    current_speaker = spk_id;
    
    Beatrice20rc0_SetCodebook(
        rc0_phone_ctx,
        codebooks.data() + static_cast<size_t>(spk_id) *
                               BEATRICE_20RC0_CODEBOOK_SIZE *
                               BEATRICE_20RC0_PHONE_CHANNELS);
    Beatrice20rc0_SetAdditiveSpeakerEmbedding(
        rc0_setter,
        spk_emb.data() + static_cast<size_t>(spk_id) *
                             BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS,
        rc0_emb_ctx, rc0_wf_ctx);
    Beatrice20rc0_RegisterKeyValueSpeakerEmbedding(
        rc0_setter,
        kv_emb.data() + static_cast<size_t>(spk_id) *
                            BEATRICE_20RC0_KV_LENGTH *
                            BEATRICE_20RC0_KV_SPEAKER_EMBEDDING_CHANNELS,
        rc0_emb_ctx);
    for (int b = 0; b < BEATRICE_20RC0_N_BLOCKS; ++b) {
      Beatrice20rc0_SetKeyValueSpeakerEmbedding(rc0_setter, b, rc0_emb_ctx, rc0_wf_ctx);
    }
  }

  void apply_rc0_formant() {
    if (!loaded || version != 2) return;
    double f = std::clamp(formant_shift, -2.0, 2.0);
    int idx = std::clamp(static_cast<int>(std::round(f * 2.0 + 4.0)), 0, 8);
    Beatrice20rc0_SetFormantShiftEmbedding(
        rc0_setter,
        formant_emb.data() + static_cast<size_t>(idx) *
                                 BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS,
        rc0_emb_ctx, rc0_wf_ctx);
  }

  void apply_pitch_bounds() {
    if (!loaded) return;
    int max_bins = (version == 2) ? BEATRICE_20RC0_PITCH_BINS : BEATRICE_20B1_PITCH_BINS;
    int min_b = std::clamp(static_cast<int>(std::round((min_src_pitch_midi - 33.0) * kBinsPerSemitone)), 1, max_bins - 1);
    int max_b = std::clamp(static_cast<int>(std::round((max_src_pitch_midi - 33.0) * kBinsPerSemitone)), min_b, max_bins - 1);
    if (version == 2) {
      Beatrice20rc0_SetMinQuantizedPitch(rc0_pitch_ctx, min_b);
      Beatrice20rc0_SetMaxQuantizedPitch(rc0_pitch_ctx, max_b);
      Beatrice20rc0_SetVQNumNeighbors(rc0_phone_ctx, std::clamp(vq_neighbors, 0, 8));
    } else if (version == 1) {
      Beatrice20b1_SetMinQuantizedPitch(b1_pitch_ctx, min_b);
      Beatrice20b1_SetMaxQuantizedPitch(b1_pitch_ctx, max_b);
    } else if (version == 0) {
      Beatrice20a2_SetMinQuantizedPitch(a2_pitch_ctx, min_b);
      Beatrice20a2_SetMaxQuantizedPitch(a2_pitch_ctx, max_b);
    }
  }

  int shift_and_floor_bin(int raw_bin, int max_bins) const {
    double avg_bin = (avg_src_pitch_midi - 33.0) * kBinsPerSemitone;
    double shifted = avg_bin + (static_cast<double>(raw_bin) - avg_bin) * intonation
                     + kBinsPerSemitone * pitch_shift;

    // soft knee floor so deep vocal fry doesnt hard-clip
    if (pitch_floor_hz > 40.0) {
      double floor_bin = hz_to_beatrice_bin(pitch_floor_hz);
      double knee_width = 4.0 * kBinsPerSemitone;
      if (shifted < floor_bin + knee_width) {
        double below = (floor_bin + knee_width) - shifted;
        shifted = floor_bin + knee_width - knee_width * std::tanh(below / knee_width);
      }
    }
    return std::clamp(static_cast<int>(std::round(shifted)), 1, max_bins - 1);
  }
};
	
  
		
  
  
extern "C" {

BeatriceBridge* br_create(void) {
  return new BeatriceBridge();
}

void br_destroy(BeatriceBridge* ctx) {
  if (!ctx) return;
  ctx->unload();
  delete ctx;
}

int br_load(BeatriceBridge* ctx, const char* dir_path, int version) {
  if (!ctx || !dir_path) return -1;
  ctx->unload();

  fs::path d(dir_path);
  auto p_phone = (d / "phone_extractor.bin").string();
  auto p_pitch = (d / "pitch_estimator.bin").string();
  auto p_wf = (d / "waveform_generator.bin").string();
  auto p_spk = (d / "speaker_embeddings.bin").string();
  
  if (version == 2) {
    auto p_set = (d / "embedding_setter.bin").string();
    ctx->rc0_phone = Beatrice20rc0_CreatePhoneExtractor();
    ctx->rc0_phone_ctx = Beatrice20rc0_CreatePhoneContext1();
    ctx->rc0_pitch = Beatrice20rc0_CreatePitchEstimator();
    ctx->rc0_pitch_ctx = Beatrice20rc0_CreatePitchContext1();
    ctx->rc0_wf = Beatrice20rc0_CreateWaveformGenerator();
    ctx->rc0_wf_ctx = Beatrice20rc0_CreateWaveformContext1();
    ctx->rc0_setter = Beatrice20rc0_CreateEmbeddingSetter();
    ctx->rc0_emb_ctx = Beatrice20rc0_CreateEmbeddingContext();

    if (Beatrice20rc0_ReadPhoneExtractorParameters(ctx->rc0_phone, p_phone.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -2; }
    if (Beatrice20rc0_ReadPitchEstimatorParameters(ctx->rc0_pitch, p_pitch.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -3; }
    if (Beatrice20rc0_ReadWaveformGeneratorParameters(ctx->rc0_wf, p_wf.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -4; }
    if (Beatrice20rc0_ReadEmbeddingSetterParameters(ctx->rc0_setter, p_set.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -5; }

    int n_spk = 0;
    if (Beatrice20rc0_ReadNSpeakers(p_spk.c_str(), &n_spk) != Beatrice_kSuccess || n_spk <= 0) { ctx->unload(); return -6; }
    ctx->n_speakers = n_spk;
    ctx->codebooks.resize(static_cast<size_t>(n_spk) * BEATRICE_20RC0_CODEBOOK_SIZE * BEATRICE_20RC0_PHONE_CHANNELS);
    ctx->spk_emb.resize(static_cast<size_t>(n_spk) * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    ctx->formant_emb.resize(9 * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    ctx->kv_emb.resize(static_cast<size_t>(n_spk) * BEATRICE_20RC0_KV_LENGTH * BEATRICE_20RC0_KV_SPEAKER_EMBEDDING_CHANNELS);

    if (Beatrice20rc0_ReadSpeakerEmbeddings(p_spk.c_str(), ctx->codebooks.data(), ctx->spk_emb.data(), ctx->formant_emb.data(), ctx->kv_emb.data()) != Beatrice_kSuccess) {
      ctx->unload();
      return -7;
    }

    ctx->version = 2;
    ctx->loaded = true;
    ctx->apply_rc0_speaker(0);
    ctx->apply_rc0_formant();
    ctx->apply_pitch_bounds();
    return n_spk;
  } else if (version == 1) {
    ctx->b1_phone = Beatrice20b1_CreatePhoneExtractor();
    ctx->b1_phone_ctx = Beatrice20b1_CreatePhoneContext1();
    ctx->b1_pitch = Beatrice20b1_CreatePitchEstimator();
    ctx->b1_pitch_ctx = Beatrice20b1_CreatePitchContext1();
    ctx->b1_wf = Beatrice20b1_CreateWaveformGenerator();
    ctx->b1_wf_ctx = Beatrice20b1_CreateWaveformContext1();

    if (Beatrice20b1_ReadPhoneExtractorParameters(ctx->b1_phone, p_phone.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -2; }
    if (Beatrice20b1_ReadPitchEstimatorParameters(ctx->b1_pitch, p_pitch.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -3; }
    if (Beatrice20b1_ReadWaveformGeneratorParameters(ctx->b1_wf, p_wf.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -4; }

    int n_spk = 0;
    if (Beatrice20b1_ReadNSpeakers(p_spk.c_str(), &n_spk) != Beatrice_kSuccess || n_spk <= 0) { ctx->unload(); return -6; }
    ctx->n_speakers = n_spk;
    ctx->spk_emb.resize(static_cast<size_t>(n_spk) * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    if (Beatrice20b1_ReadSpeakerEmbeddings(p_spk.c_str(), ctx->spk_emb.data()) != Beatrice_kSuccess) { ctx->unload(); return -7; }

    ctx->version = 1;
    ctx->current_speaker = 0;
    ctx->loaded = true;
    ctx->apply_pitch_bounds();
    return n_spk;
  } else if (version == 0) {
    ctx->a2_phone = Beatrice20a2_CreatePhoneExtractor();
    ctx->a2_phone_ctx = Beatrice20a2_CreatePhoneContext1();
    ctx->a2_pitch = Beatrice20a2_CreatePitchEstimator();
    ctx->a2_pitch_ctx = Beatrice20a2_CreatePitchContext1();
    ctx->a2_wf = Beatrice20a2_CreateWaveformGenerator();
    ctx->a2_wf_ctx = Beatrice20a2_CreateWaveformContext1();

    if (Beatrice20a2_ReadPhoneExtractorParameters(ctx->a2_phone, p_phone.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -2; }
    if (Beatrice20a2_ReadPitchEstimatorParameters(ctx->a2_pitch, p_pitch.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -3; }
    if (Beatrice20a2_ReadWaveformGeneratorParameters(ctx->a2_wf, p_wf.c_str()) != Beatrice_kSuccess) { ctx->unload(); return -4; }

    int n_spk = 0;
    if (Beatrice20a2_ReadNSpeakers(p_spk.c_str(), &n_spk) != Beatrice_kSuccess || n_spk <= 0) { ctx->unload(); return -6; }
    ctx->n_speakers = n_spk;
    ctx->spk_emb.resize(static_cast<size_t>(n_spk) * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    if (Beatrice20a2_ReadSpeakerEmbeddings(p_spk.c_str(), ctx->spk_emb.data()) != Beatrice_kSuccess) { ctx->unload(); return -7; }

    ctx->version = 0;
    ctx->current_speaker = 0;
    ctx->loaded = true;
    ctx->apply_pitch_bounds();
    return n_spk;
  }
  return -10;
}

void br_set_speaker(BeatriceBridge* ctx, int speaker_id) {
  if (!ctx || !ctx->loaded || ctx->n_speakers <= 0) return;
  speaker_id = std::clamp(speaker_id, 0, ctx->n_speakers - 1);
  if (ctx->version == 2) {
    ctx->apply_rc0_speaker(speaker_id);
  } else {
    ctx->current_speaker = speaker_id;
  }
}

void br_set_params(
    BeatriceBridge* ctx,
    double pitch_shift,
    double formant_shift,
    double intonation,
    int vq_neighbors,
    double avg_src_pitch_midi,
    double min_src_pitch_midi,
    double max_src_pitch_midi,
    double pitch_floor_hz) {
  if (!ctx) return;
  ctx->pitch_shift = std::clamp(pitch_shift, -24.0, 24.0);
  ctx->formant_shift = std::clamp(formant_shift, -2.0, 2.0);
  ctx->intonation = std::clamp(intonation, 0.0, 2.0);
  ctx->vq_neighbors = std::clamp(vq_neighbors, 0, 8);
  ctx->avg_src_pitch_midi = std::clamp(avg_src_pitch_midi, 24.0, 96.0);
  ctx->min_src_pitch_midi = std::clamp(min_src_pitch_midi, 24.0, 84.0);
  ctx->max_src_pitch_midi = std::clamp(max_src_pitch_midi, 36.0, 108.0);
  ctx->pitch_floor_hz = std::max(0.0, pitch_floor_hz);

  if (ctx->loaded) {
    ctx->apply_rc0_formant();
    ctx->apply_pitch_bounds();
  }
}
	
  
    
  
void br_reset(BeatriceBridge* ctx) {
  if (!ctx || !ctx->loaded) return;
  if (ctx->version == 2) {
    Beatrice20rc0_DestroyPhoneContext1(ctx->rc0_phone_ctx);
    Beatrice20rc0_DestroyPitchContext1(ctx->rc0_pitch_ctx);
    Beatrice20rc0_DestroyWaveformContext1(ctx->rc0_wf_ctx);
    Beatrice20rc0_DestroyEmbeddingContext(ctx->rc0_emb_ctx);
    ctx->rc0_phone_ctx = Beatrice20rc0_CreatePhoneContext1();
    ctx->rc0_pitch_ctx = Beatrice20rc0_CreatePitchContext1();
    ctx->rc0_wf_ctx = Beatrice20rc0_CreateWaveformContext1();
    ctx->rc0_emb_ctx = Beatrice20rc0_CreateEmbeddingContext();
    ctx->apply_rc0_speaker(ctx->current_speaker);
    ctx->apply_rc0_formant();
    ctx->apply_pitch_bounds();
  } else if (ctx->version == 1) {
    Beatrice20b1_DestroyPhoneContext1(ctx->b1_phone_ctx);
    Beatrice20b1_DestroyPitchContext1(ctx->b1_pitch_ctx);
    Beatrice20b1_DestroyWaveformContext1(ctx->b1_wf_ctx);
    ctx->b1_phone_ctx = Beatrice20b1_CreatePhoneContext1();
    ctx->b1_pitch_ctx = Beatrice20b1_CreatePitchContext1();
    ctx->b1_wf_ctx = Beatrice20b1_CreateWaveformContext1();
    ctx->apply_pitch_bounds();
  } else if (ctx->version == 0) {
    Beatrice20a2_DestroyPhoneContext1(ctx->a2_phone_ctx);
    Beatrice20a2_DestroyPitchContext1(ctx->a2_pitch_ctx);
    Beatrice20a2_DestroyWaveformContext1(ctx->a2_wf_ctx);
    ctx->a2_phone_ctx = Beatrice20a2_CreatePhoneContext1();
    ctx->a2_pitch_ctx = Beatrice20a2_CreatePitchContext1();
    ctx->a2_wf_ctx = Beatrice20a2_CreateWaveformContext1();
    ctx->apply_pitch_bounds();
  }
}

void br_process_hops(
    BeatriceBridge* ctx,
    const float* in_16k,
    float* out_24k,
    int n_hops) {
  if (!ctx || !ctx->loaded || !in_16k || !out_24k || n_hops <= 0) {
    if (out_24k && n_hops > 0) {
      std::memset(out_24k, 0, static_cast<size_t>(n_hops) * BEATRICE_OUT_HOP_LENGTH * sizeof(float));
    }
    return;
  }

  if (ctx->version == 2) {
    std::array<float, BEATRICE_20RC0_PHONE_CHANNELS> phone;
    std::array<float, 4> pitch_feat;
    for (int h = 0; h < n_hops; ++h) {
      const float* x = in_16k + static_cast<size_t>(h) * BEATRICE_IN_HOP_LENGTH;
      float* y = out_24k + static_cast<size_t>(h) * BEATRICE_OUT_HOP_LENGTH;
      Beatrice20rc0_ExtractPhone1(ctx->rc0_phone, x, phone.data(), ctx->rc0_phone_ctx);
      int qpitch = 1;
      Beatrice20rc0_EstimatePitch1(ctx->rc0_pitch, x, &qpitch, pitch_feat.data(), ctx->rc0_pitch_ctx);
      qpitch = ctx->shift_and_floor_bin(qpitch, BEATRICE_20RC0_PITCH_BINS);
      Beatrice20rc0_GenerateWaveform1(ctx->rc0_wf, phone.data(), &qpitch, pitch_feat.data(), y, ctx->rc0_wf_ctx);
    }
  } else if (ctx->version == 1) {
    std::array<float, BEATRICE_20B1_PHONE_CHANNELS> phone;
    std::array<float, 4> pitch_feat;
    const float* spk = ctx->spk_emb.data() + static_cast<size_t>(ctx->current_speaker) * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS;
    for (int h = 0; h < n_hops; ++h) {
      const float* x = in_16k + static_cast<size_t>(h) * BEATRICE_IN_HOP_LENGTH;
      float* y = out_24k + static_cast<size_t>(h) * BEATRICE_OUT_HOP_LENGTH;
      Beatrice20b1_ExtractPhone1(ctx->b1_phone, x, phone.data(), ctx->b1_phone_ctx);
      int qpitch = 1;
      Beatrice20b1_EstimatePitch1(ctx->b1_pitch, x, &qpitch, pitch_feat.data(), ctx->b1_pitch_ctx);
      qpitch = ctx->shift_and_floor_bin(qpitch, BEATRICE_20B1_PITCH_BINS);
      Beatrice20b1_GenerateWaveform1(ctx->b1_wf, phone.data(), &qpitch, pitch_feat.data(), spk, y, ctx->b1_wf_ctx);
    }
  } else if (ctx->version == 0) {
    std::array<float, BEATRICE_20A2_PHONE_CHANNELS> phone;
    std::array<float, 4> pitch_feat;
    const float* spk = ctx->spk_emb.data() + static_cast<size_t>(ctx->current_speaker) * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS;
    for (int h = 0; h < n_hops; ++h) {
      const float* x = in_16k + static_cast<size_t>(h) * BEATRICE_IN_HOP_LENGTH;
      float* y = out_24k + static_cast<size_t>(h) * BEATRICE_OUT_HOP_LENGTH;
      Beatrice20a2_ExtractPhone1(ctx->a2_phone, x, phone.data(), ctx->a2_phone_ctx);
      int qpitch = 1;
      Beatrice20a2_EstimatePitch1(ctx->a2_pitch, x, &qpitch, pitch_feat.data(), ctx->a2_pitch_ctx);
      qpitch = ctx->shift_and_floor_bin(qpitch, BEATRICE_20A2_PITCH_BINS);
      Beatrice20a2_GenerateWaveform1(ctx->a2_wf, phone.data(), &qpitch, pitch_feat.data(), spk, y, ctx->a2_wf_ctx);
    }
  }
}

}
