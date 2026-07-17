// 進捗報告スライド（3週間まとめ） 2026-07-17
// コンパイル: typst compile 2026_07_17_progress_ja.typ
// フォント: CJK が出ない場合は下の font を各自の環境にあるものへ変更してください
//   例: "Hiragino Sans" / "Yu Gothic" / "Noto Sans CJK JP" / "IPAexGothic"

#set page(width: 25cm, height: 14.06cm, margin: (x: 1.8cm, top: 1.5cm, bottom: 1.1cm))
#set text(font: ("Noto Sans CJK JP", "Noto Sans CJK SC", "Hiragino Sans"), size: 19pt, lang: "ja")
#set par(leading: 0.72em)
#let accent = rgb("#1f4e79")
#let good = rgb("#1b7f4d")
#let bad = rgb("#a23b2c")

#let slide(title, body) = {
  block(width: 100%, breakable: false)[
    #text(size: 25pt, weight: "bold", fill: accent)[#title]
    #v(0.15em)
    #line(length: 100%, stroke: 1.1pt + accent)
    #v(0.55em)
    #set text(size: 19pt)
    #body
  ]
  pagebreak(weak: true)
}

// ---- タイトルスライド ----
#align(center + horizon)[
  #text(size: 33pt, weight: "bold", fill: accent)[視覚ドッキング: SRL + DRL 進捗報告]
  #v(0.8em)
  #text(size: 21pt)[自己教師あり表現（SPR）+ 強化学習方策]
  #v(1.6em)
  #text(size: 16pt, fill: gray)[3週間まとめ ・ 2026-07-17]
]
#pagebreak()

#slide[研究の位置づけ・目標][
- タスク: 視覚 student が前方のボールへ *整列 → 接近 → 停止*（ドッキング）
- 枠組み: *SRL + DRL* — 自己教師あり表現（SPR）の上に強化学習の方策を載せる
- 特権情報なし・*視覚のみ*（RGB 単眼）
- 長期目標: レーダ/深度に頼らない、人の感覚範囲での自律ナビへの一歩
- 現タスクはその最小抽象（目標固定・単純シーン）
]

#slide[前2週の要点 — SPR-PPO（on-policy）][
- SPR 表現 + PPO（on-policy）で方策を学習
- 逆カリキュラム JSRL: best success *62.5%*（de = 1 m, xe = 20 px）
- 純 SPR-PPO（カリキュラムなし）: *〜0%*（28% でピーク → 崩壊）
- 症状: 「区域まで行く」は解けるが、*精密な停止・整列が学べない*
- #text(fill: bad)[→ on-policy では頭打ち]
]

#slide[なぜ on-policy がダメか (1)：σ が潰れない][
- success 条件: 3次元の行動が同時に `|a_i| < 0.05`
- 成功確率:
#align(center)[$ P("success") approx [ 2 Phi(0.05 / sigma) - 1 ]^3 $]
- → σ が十分小さくないと成功しない
- しかし PPO の `log_std` 勾配は*疎な終端報酬から信号を得られず*、σ が下限クランプに張り付く（特に a_x「制動」・a_w「整列」次元）
- 決定論評価は μ を使うが、μ 自体も精密化されない
]

#slide[なぜ on-policy がダメか (2)：データ効率][
- on-policy は*毎更新でデータを破棄* → 稀な成功を再利用できない
- 疎な `+10`（成功）報酬を接近軌道に沿って*伝播しきれない*
- 報酬設計上、stop 区の shaping は OFF、`d_gain (0.0018) < K_TIME (0.003)` で
  「接近」はむしろ罰 → 唯一効くのは終端 `+10`
- #text(fill: good)[off-policy は replay で稀な成功を反復再生 → credit assignment が通る]
]

#slide[off-policy（DrQ-v2）へ切替 → いずれも良好][
#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, left),
  stroke: 0.5pt + gray,
  [*設定*], [*success*], [*精度*],
  [arm A（凍結 SPR z + DrQ-v2）], text(fill: good)[*100%*], [de 0.022 m / xe 0.9 px],
  [pixels（クリーン env）], text(fill: good)[*97.7%*], [—],
  [pixels（ノイズ env）], text(fill: good)[*99.2%*], [—],
)
#v(0.5em)
- 動きがクリーン: *加速 → 一段制動 → 停止*、ブレなし
- σ は「学習」ではなく「スケジュール」（0.3 → 0.05）で下げる
]

#slide[表現は十分 — 失敗はアルゴリズム][
- *同じ凍結 SPR z* が off-policy で 100% → 表現に情報は足りている
- critic 適合度（割引リターン回帰 R²）: #text(fill: good)[*SPR z 0.84 > oracle 0.78*]
- TD(0) は全表現で失敗、n = 17（GAE 有効視野）で回復
- #text(fill: accent)[結論: 「表現が悪い」ではなく「on-policy のアルゴリズム（信用割当）」が原因]
- 同一 z・唯一の変数はアルゴリズム: PPO 0〜28% vs off-policy 100%
]

#slide[現在の SRL + DRL パイプライン][
+ *データ収集*: teacher で軌道を収集（視覚 + 特権状態）
+ *SPR 事前学習*: 自己予測表現 z を学習
  - ResNet18 + FPN encoder、EMA target、行動条件付き転移予測
+ *encoder を凍結*（BN ドリフト回避のため常に eval）
+ *off-policy 学習*: DrQ-v2 が z 上で方策を学習（評価は決定論的 μ）
]

#slide[次の方向][
- *世界モデル線*: 凍結/共学習した SPR latent 上で *Dreamer / TD-MPC*
  - model-based off-policy = サンプル効率の天井
  - 実機・少サンプル・逐次適応（次のフェーズ）に直結
  - 「感知 + 決定」を1つの学習対象に統合できる
- on-policy は「*replay なしで疎な終端信用を精度次元へ送る*」機構
  （報酬再分配 / per-dim σ 制御 など）が見つかれば再訪
]
