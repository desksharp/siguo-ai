// 碰子规则、布阵约束与布阵生成、随机数
#include <algorithm>
#include <array>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "junqi/junqi.h"

namespace junqi {

#if defined(_MSC_VER)
int Popcount(Mask m) { return int(__popcnt(unsigned(m))); }
#else
int Popcount(Mask m) { return __builtin_popcount(unsigned(m)); }
#endif

uint64_t SplitMix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void Rng::Seed(uint64_t seed) {
  uint64_t x = seed;
  for (auto& v : s) v = SplitMix64(x);
}

uint64_t Rng::Next() {  // xoshiro256**
  auto rotl = [](uint64_t v, int k) { return (v << k) | (v >> (64 - k)); };
  const uint64_t result = rotl(s[1] * 5, 7) * 9;
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

int Resolve(int a, int d) {
  if (a < 0 || d < 0 || a == kFlag || a == kMine) return kOutIllegal;
  if (d == kFlag) return kOutCaptureFlag;
  if (a == kBomb || d == kBomb) return kOutBoth;
  if (d == kMine) return a == kEngineer ? kOutKill : kOutKilled;
  if (a > d) return kOutKill;
  if (a < d) return kOutKilled;
  return kOutBoth;
}

namespace {
struct CompatTables {
  Mask def[5][kTypes] = {};
  Mask att[5][kTypes] = {};
  CompatTables() {
    for (int a = 0; a < kTypes; ++a)
      for (int d = 0; d < kTypes; ++d) {
        const int o = Resolve(a, d);
        if (o < 0) continue;
        def[o][a] |= Bit(d);
        att[o][d] |= Bit(a);
      }
  }
};
const CompatTables& Compat() {
  static const CompatTables t;
  return t;
}
}  // namespace

Mask CompatDefender(int outcome, int attacker_type) {
  if (outcome < 0 || outcome > 4 || attacker_type < 0) return 0;
  return Compat().def[outcome][attacker_type];
}
Mask CompatAttacker(int outcome, int defender_type) {
  if (outcome < 0 || outcome > 4 || defender_type < 0) return 0;
  return Compat().att[outcome][defender_type];
}

bool ValidSetup(const int8_t* camp30) {
  std::array<int, kTypes> cnt{};
  for (int pos = 0; pos < kCampNodes; ++pos) {
    const int r = pos / 5, c = pos % 5, t = camp30[pos];
    if (IsCampCell(r, c)) {
      if (t != -1) return false;
      continue;
    }
    if (t < 0 || t >= kTypes) return false;
    if (t == kFlag && !IsBaseCell(r, c)) return false;
    if (t == kMine && r < 4) return false;
    if (t == kBomb && r == 0) return false;
    ++cnt[t];
  }
  for (int t = 0; t < kTypes; ++t)
    if (cnt[t] != kQuota[t]) return false;
  return true;
}

namespace {
// 旧版布阵（kSetupLegacy）：按兵种分组的行权重（行 0=前排 … 5=底排）
const float kRowWeight[6][6] = {
    {3.0f, 2.5f, 2.0f, 1.5f, 1.0f, 0.3f},  // 司令 军长
    {2.5f, 2.0f, 2.0f, 1.5f, 1.0f, 0.5f},  // 师长 旅长
    {2.0f, 1.5f, 1.5f, 1.5f, 1.0f, 0.7f},  // 团长 营长
    {2.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.2f},  // 连长 排长
    {1.5f, 2.0f, 1.5f, 1.5f, 1.0f, 0.6f},  // 工兵
    {0.0f, 2.0f, 2.0f, 1.5f, 1.0f, 0.3f},  // 炸弹
};
int Group(int t) {
  switch (t) {
    case kMarshal:
    case kCorps:
      return 0;
    case kDivision:
    case kBrigade:
      return 1;
    case kRegiment:
    case kBattalion:
      return 2;
    case kCompany:
    case kPlatoon:
      return 3;
    case kEngineer:
      return 4;
    default:
      return 5;
  }
}

template <class W>
int PickPos(const int8_t* camp30, Rng& rng, W&& weight) {
  float w[kCampNodes];
  float total = 0.0f;
  for (int pos = 0; pos < kCampNodes; ++pos) {
    const int r = pos / 5, c = pos % 5;
    w[pos] = (IsCampCell(r, c) || camp30[pos] != -1) ? 0.0f : weight(r, c);
    total += w[pos];
  }
  float u = float(rng.Uniform()) * total;
  int last = -1;
  for (int pos = 0; pos < kCampNodes; ++pos) {
    if (w[pos] <= 0.0f) continue;
    last = pos;
    if (u < w[pos]) return pos;
    u -= w[pos];
  }
  return last;
}
}  // namespace

static float g_setup_params[kSetupSlots][kSetupParams];

void SetSetupParams(int slot, const float* params) {
  if (slot >= 0 && slot < kSetupSlots) std::copy(params, params + kSetupParams, g_setup_params[slot]);
}

// 现行布阵（kSetupHeuristic）：setup_search 搜出来的参数。参数含义见 GenerateCustomSetup。
// 候选一队对上一版启发式：rl_four_v5（两种布阵各训练一半）0.603 ± 0.030，1000 局
const float kSearchedSetup[kSetupParams] = {
    0.07211f, 0.0729f, 0.597f,  // 另一个大本营：雷 排长 连长
    25.3f, 7.712f, 0.3134f, 2.159f, 0.4481f, 9.007f, 2.431f, 1.522f,  // 地雷：第 4 行五列、第 5 行三列（军旗在第 1 列）
    0.0f, 10.94f, 1.783f, 0.3632f, 0.2729f, 0.06933f,  // 炸弹：行权重
    2.154f, 0.4438f, 0.8039f, 8.22f, 1.57f, 0.0f,  // 工兵：行权重
    6.557f, 2.013f, 0.3987f, 1.608f, 0.9816f, 0.4121f,  // 排长：行权重
    2.664f, 0.942f, 0.2119f, 1.092f, 0.8828f, 4.578f,  // 连长：行权重
    0.5804f, 0.8409f, 0.2915f, 1.422f, 3.279f, 0.9168f,  // 营长：行权重
    1.315f, 1.503f, 2.755f, 2.556f, 2.196f, 1.205f,  // 团长：行权重
    1.977f, 2.927f, 1.673f, 1.434f, 0.8144f, 0.7307f,  // 旅长：行权重
    1.106f, 3.966f, 2.059f, 0.4293f, 1.303f, 0.7266f,  // 师长：行权重
    3.586f, 2.728f, 1.076f, 26.76f, 1.845f, 0.04445f,  // 军长：行权重
    1.746f, 2.95f, 0.5225f, 2.381f, 0.276f, 1.151f,  // 司令：行权重
    2.77f, 0.1501f, 0.4877f, 0.3335f, 4.59f,  // 炸弹：列权重
    0.8527f, 1.248f, 0.2542f, 7.366f, 3.805f,  // 工兵：列权重
    0.4935f, 2.547f, 0.2218f, 1.711f, 0.2745f,  // 排长：列权重
    3.075f, 0.2798f, 1.87f, 1.59f, 0.8861f,  // 连长：列权重
    1.418f, 0.6887f, 0.7013f, 0.1892f, 1.43f,  // 营长：列权重
    1.204f, 0.6043f, 0.8107f, 0.6779f, 5.447f,  // 团长：列权重
    1.806f, 0.9423f, 0.3129f, 0.2142f, 0.9544f,  // 旅长：列权重
    1.978f, 0.4959f, 0.159f, 0.5216f, 1.559f,  // 师长：列权重
    0.5844f, 0.1089f, 1.117f, 0.6889f, 0.2764f,  // 军长：列权重
    1.49f, 0.9574f, 0.1496f, 0.4627f, 1.254f,  // 司令：列权重
};

static void GenerateCustomSetup(const float* P, Rng& rng, int8_t* camp30) {
  const float* hq = P, *mine = P + 3, *row = P + 11, *col = P + 71;
  const int flag_col = rng.Below(2) == 0 ? 1 : 3;
  const auto canon = [&](int c) { return flag_col == 1 ? c : 4 - c; };
  camp30[5 * 5 + flag_col] = kFlag;
  const float u = float(rng.Uniform()) * (hq[0] + hq[1] + hq[2]);
  const int hq_type = u < hq[0] ? kMine : u < hq[0] + hq[1] ? kPlatoon : kCompany;
  camp30[5 * 5 + 4 - flag_col] = int8_t(hq_type);
  for (int k = hq_type == kMine ? 1 : 0; k < kQuota[kMine]; ++k) {
    const int pos = PickPos(camp30, rng, [&](int r, int c) -> float {
      if (r < 4) return 0.0f;
      const int cc = canon(c);
      return r == 4 ? mine[cc] : mine[5 + cc / 2];  // 第 5 行非大本营是第 0、2、4 列
    });
    camp30[pos] = kMine;
  }
  std::vector<int> order = {kBomb, kBomb, kMarshal, kCorps, kDivision, kDivision, kBrigade, kBrigade,
                            kRegiment, kRegiment, kBattalion, kBattalion, kEngineer, kEngineer, kEngineer,
                            kCompany, kCompany, kCompany, kPlatoon, kPlatoon, kPlatoon};
  if (hq_type != kMine) order.erase(std::find(order.begin(), order.end(), hq_type));
  for (int t : order) {
    const int pos = PickPos(camp30, rng, [&](int r, int c) -> float {
      if (t == kBomb && r == 0) return 0.0f;
      return row[(t - kBomb) * 6 + r] * col[(t - kBomb) * 5 + canon(c)];
    });
    camp30[pos] = int8_t(t);
  }
}

void GenerateSetup(int kind, Rng& rng, int8_t* camp30) {
  for (int pos = 0; pos < kCampNodes; ++pos) camp30[pos] = -1;
  if (kind >= kSetupCustom && kind < kSetupCustom + kSetupSlots)
    return GenerateCustomSetup(g_setup_params[kind - kSetupCustom], rng, camp30);
  if (kind == kSetupHeuristic) return GenerateCustomSetup(kSearchedSetup, rng, camp30);
  const bool heur = kind != kSetupRandom;
  const int flag_col = rng.Below(2) == 0 ? 1 : 3;
  camp30[5 * 5 + flag_col] = kFlag;
  for (int k = 0; k < kQuota[kMine]; ++k) {
    const int pos = PickPos(camp30, rng, [&](int r, int c) -> float {
      if (r < 4) return 0.0f;
      if (!heur) return 1.0f;
      const bool guard = (r == 5 && (c == flag_col - 1 || c == flag_col + 1)) || (r == 4 && c == flag_col);
      return guard ? 6.0f : (r == 5 ? 2.0f : 1.5f);
    });
    camp30[pos] = kMine;
  }
  std::vector<int> order = {kBomb, kBomb, kMarshal, kCorps, kDivision, kDivision, kBrigade, kBrigade,
                            kRegiment, kRegiment, kBattalion, kBattalion, kEngineer, kEngineer, kEngineer,
                            kCompany, kCompany, kCompany, kPlatoon, kPlatoon, kPlatoon};
  for (int t : order) {
    const int pos = PickPos(camp30, rng, [&](int r, int) -> float {
      if (t == kBomb && r == 0) return 0.0f;
      return heur ? kRowWeight[Group(t)][r] : 1.0f;
    });
    camp30[pos] = int8_t(t);
  }
}

}  // namespace junqi
