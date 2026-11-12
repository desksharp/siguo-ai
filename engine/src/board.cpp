// 棋盘图：节点、公路/铁路边、直行连续表、座位旋转、距离表
#include <cassert>
#include <deque>
#include <mutex>
#include <utility>

#include "junqi/junqi.h"

namespace junqi {
namespace {

// 17×17 网格坐标，旋转 90°：(x, y) → (y, 16-x)，把 seat s 的营区映射到 seat s+1
void Rot(int& x, int& y) {
  const int nx = y, ny = 16 - x;
  x = nx;
  y = ny;
}
void RotN(int& x, int& y, int n) {
  for (int k = 0; k < (n % 4 + 4) % 4; ++k) Rot(x, y);
}
void CampXY(int seat, int r, int c, int& x, int& y) {
  x = 6 + c;
  y = 11 + r;
  RotN(x, y, seat);
}

}  // namespace

Board::Board(int arc_mode_in, bool center_middle_rail_in, bool camp_corner_straight_in)
    : arc_mode(arc_mode_in), center_middle_rail(center_middle_rail_in), camp_corner_straight(camp_corner_straight_in) {
  node_at.fill(-1);
  for (int s = 0; s < kSeats; ++s) {
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 5; ++c) {
        const int n = CampNode(s, r, c);
        int px, py;
        CampXY(s, r, c, px, py);
        seat_of[n] = int8_t(s);
        row[n] = int8_t(r);
        col[n] = int8_t(c);
        x[n] = int8_t(px);
        y[n] = int8_t(py);
        is_camp[n] = IsCampCell(r, c);
        is_base[n] = IsBaseCell(r, c);
        node_at[py * kGrid + px] = int16_t(n);
      }
    }
  }
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      const int n = kCenterBase + j * 3 + i;
      seat_of[n] = -1;
      row[n] = int8_t(j);
      col[n] = int8_t(i);
      x[n] = int8_t(6 + 2 * i);
      y[n] = int8_t(6 + 2 * j);
      node_at[y[n] * kGrid + x[n]] = int16_t(n);
    }
  }
  auto at = [&](int px, int py) {
    const int n = node_at[py * kGrid + px];
    assert(n >= 0);
    return n;
  };
  auto camp = [&](int s, int r, int c) { return CampNode(((s % 4) + 4) % 4, r, c); };
  auto center_rot = [&](int px, int py, int s) {
    RotN(px, py, s);
    return at(px, py);
  };

  auto add_edge = [&](int a, int b, int kind) {
    if (edge_kind[a][b] == 0) {
      adj[a].push_back(int16_t(b));
      adj[b].push_back(int16_t(a));
    }
    if (kind > edge_kind[a][b]) edge_kind[a][b] = edge_kind[b][a] = uint8_t(kind);
  };

  // ---- 直线铁路（非循环），连续三点视为“直行”
  std::vector<std::vector<int>> lines;
  for (int s = 0; s < kSeats; ++s) {
    std::vector<int> front, row4;
    for (int c = 0; c < 5; ++c) {
      front.push_back(camp(s, 0, c));
      row4.push_back(camp(s, 4, c));
    }
    lines.push_back(front);
    lines.push_back(row4);
  }
  for (int s = 0; s < 2; ++s) {  // 南北贯通线（s=0）与东西贯通线（s=1）
    std::vector<int> a, b, m;
    for (int r = 4; r >= 0; --r) a.push_back(camp(s, r, 0));
    for (int py : {10, 8, 6}) a.push_back(center_rot(6, py, s));
    for (int r = 0; r <= 4; ++r) a.push_back(camp(s + 2, r, 4));
    for (int r = 4; r >= 0; --r) b.push_back(camp(s, r, 4));
    for (int py : {10, 8, 6}) b.push_back(center_rot(10, py, s));
    for (int r = 0; r <= 4; ++r) b.push_back(camp(s + 2, r, 0));
    if (center_middle_rail) m.push_back(camp(s, 0, 2));
    for (int py : {10, 8, 6}) m.push_back(center_rot(8, py, s));
    if (center_middle_rail) m.push_back(camp(s + 2, 0, 2));
    lines.push_back(a);
    lines.push_back(b);
    lines.push_back(m);
  }
  for (const auto& L : lines)
    for (size_t k = 0; k + 1 < L.size(); ++k) add_edge(L[k], L[k + 1], 2);

  // ---- 弧形铁路：seat s 前排右角 ↔ seat s+1 前排左角
  struct Arc {
    int k, k2;
  };
  std::vector<Arc> arcs;
  for (int s = 0; s < kSeats; ++s) {
    arcs.push_back({camp(s, 0, 4), camp(s + 1, 0, 0)});
    add_edge(arcs.back().k, arcs.back().k2, 2);
  }

  // ---- 营区公路与行营斜线
  for (int s = 0; s < kSeats; ++s) {
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 5; ++c) {
        if (c < 4) add_edge(camp(s, r, c), camp(s, r, c + 1), 1);
        if (r < 5) add_edge(camp(s, r, c), camp(s, r + 1, c), 1);
        if (IsCampCell(r, c)) {
          for (int dr : {-1, 1})
            for (int dc : {-1, 1}) {
              const int rr = r + dr, cc = c + dc;
              if (rr >= 0 && rr < 6 && cc >= 0 && cc < 5) add_edge(camp(s, r, c), camp(s, rr, cc), 1);
            }
        }
      }
    }
    if (!center_middle_rail) add_edge(camp(s, 0, 2), center_rot(8, 10, s), 1);
  }

  for (int n = 0; n < kNodes; ++n) {
    for (int16_t m : adj[n])
      if (edge_kind[n][m] == 2) rail_adj[n].push_back(m);
    is_rail[n] = !rail_adj[n].empty();
    straight[n].assign(rail_adj[n].size(), 0);
    for (auto& v : rail_index[n]) v = -1;
    for (size_t i = 0; i < rail_adj[n].size(); ++i) rail_index[n][rail_adj[n][i]] = int8_t(i);
    assert(rail_adj[n].size() <= 6);
  }
  auto add_straight = [&](int a, int b, int c) {  // a→b→c 直行（双向）
    straight[b][rail_index[b][a]] |= uint8_t(1u << rail_index[b][c]);
    straight[b][rail_index[b][c]] |= uint8_t(1u << rail_index[b][a]);
  };
  for (const auto& L : lines)
    for (size_t k = 0; k + 2 < L.size(); ++k) add_straight(L[k], L[k + 1], L[k + 2]);
  for (int s = 0; s < kSeats; ++s) {
    const int k = arcs[s].k, k2 = arcs[s].k2;
    const int p_front = camp(s, 0, 3), p_side = camp(s, 1, 4);
    const int n_front = camp(s + 1, 0, 1), n_side = camp(s + 1, 1, 0);
    std::vector<int> entries, exits;
    if (arc_mode == kArcFrontRows || arc_mode == kArcBoth) {
      entries.push_back(p_front);
      exits.push_back(n_front);
    }
    if (arc_mode == kArcSideRails || arc_mode == kArcBoth) {
      entries.push_back(p_side);
      exits.push_back(n_side);
    }
    for (int p : entries) add_straight(p, k, k2);
    for (int q : exits) add_straight(k, k2, q);
    if (camp_corner_straight) {  // 第5排两端的直角拐角
      add_straight(camp(s, 3, 0), camp(s, 4, 0), camp(s, 4, 1));
      add_straight(camp(s, 3, 4), camp(s, 4, 4), camp(s, 4, 3));
    }
  }

  // ---- 座位旋转表
  for (int o = 0; o < kSeats; ++o) {
    for (int n = 0; n < kNodes; ++n) {
      int r;
      if (seat_of[n] >= 0) {
        r = CampNode((seat_of[n] - o + 4) % 4, row[n], col[n]);
      } else {
        int px = x[n], py = y[n];
        RotN(px, py, -o);
        r = at(px, py);
      }
      rel[o][n] = int16_t(r);
      abs_of[o][r] = int16_t(n);
    }
  }

  // ---- 距离表（BFS，忽略占用）
  for (int s0 = 0; s0 < kNodes; ++s0) {
    for (auto& d : dist[s0]) d = 255;
    std::deque<int> q{s0};
    dist[s0][s0] = 0;
    while (!q.empty()) {
      const int a = q.front();
      q.pop_front();
      for (int16_t b : adj[a]) {
        if (dist[s0][b] != 255) continue;
        dist[s0][b] = uint8_t(dist[s0][a] + 1);
        q.push_back(b);
      }
    }
  }
}

const Board& Board::Get(int arc_mode, bool center_middle_rail, bool camp_corner_straight) {
  static Board* boards[4][2][2] = {};
  static std::once_flag once;
  std::call_once(once, [] {
    for (int a = 0; a < 4; ++a)
      for (int m = 0; m < 2; ++m)
        for (int c = 0; c < 2; ++c) boards[a][m][c] = new Board(a, m != 0, c != 0);
  });
  if (arc_mode < 0 || arc_mode > 3) arc_mode = kArcSideRails;
  return *boards[arc_mode][center_middle_rail ? 1 : 0][camp_corner_straight ? 1 : 0];
}

}  // namespace junqi
