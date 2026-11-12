// 性能基准：单线程/多线程自对弈步数每秒，以及走法生成与观测编码耗时
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "junqi/junqi.h"

using namespace junqi;
using Clock = std::chrono::steady_clock;

static double Secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

static void Bench(const char* name, int mode, int agent, int games, int threads) {
  SelfPlaySpec spec;
  spec.cfg.mode = mode;
  for (int s = 0; s < kSeats; ++s) spec.agent_kind[s] = agent;
  int64_t st[4];
  const auto t0 = Clock::now();
  RunSelfPlay(spec, games, 1, threads, nullptr, st);
  const double dt = Secs(t0);
  std::printf("%-28s threads=%2d games=%6d plies=%9lld  %8.0f plies/s  %7.1f games/s  avg_len=%.0f  W0/W1/D=%lld/%lld/%lld\n",
              name, threads, games, (long long)st[3], st[3] / dt, games / dt, double(st[3]) / games,
              (long long)st[0], (long long)st[1], (long long)st[2]);
}

int main(int argc, char** argv) {
  const int hw = int(std::thread::hardware_concurrency());
  const int threads = argc > 1 ? std::atoi(argv[1]) : hw;
  Bench("4p random", kModeFour, kAgentRandom, 2000, 1);
  Bench("4p random", kModeFour, kAgentRandom, 20000, threads);
  Bench("4p rule", kModeFour, kAgentRule, 200, 1);
  Bench("4p rule", kModeFour, kAgentRule, 2000, threads);
  Bench("duel random", kModeDuel, kAgentRandom, 2000, 1);
  Bench("duel rule", kModeDuel, kAgentRule, 400, 1);

  // 走法生成与编码
  Game g;
  g.Reset(3);
  Move buf[1024];
  std::vector<float> nf(kNodes * kNodeFeatures), gf(kGlobalFeatures);
  std::vector<uint8_t> legal(kNodes * kNodes);
  const int iters = 200000;
  auto t0 = Clock::now();
  long long total = 0;
  for (int i = 0; i < iters; ++i) total += g.LegalMoves(buf, 1024);
  const double gen = Secs(t0);
  t0 = Clock::now();
  for (int i = 0; i < iters / 10; ++i) EncodeObservation(g, i % 4, nf.data(), gf.data(), legal.data());
  const double enc = Secs(t0);
  std::printf("movegen: %.2f us/call (%lld moves)   encode: %.2f us/obs\n", gen / iters * 1e6, total / iters,
              enc / (iters / 10) * 1e6);
  return 0;
}
