// 論文紹介スライド: SPR (Self-Predictive Representations) 2026-07-17
// コンパイル: typst compile 2026_07_17_spr_paper_ja.typ
// フォント: CJK が出ない場合は下の font を各自の環境にあるものへ変更してください
//   例: "Hiragino Sans" / "Yu Gothic" / "Noto Sans CJK JP" / "IPAexGothic"

#set page(width: 25cm, height: 14.06cm, margin: (x: 1.8cm, top: 1.5cm, bottom: 1.1cm))
#set text(font: ("Noto Sans CJK JP", "Noto Sans CJK SC", "Hiragino Sans"), size: 19pt, lang: "ja")
#set par(leading: 0.72em)
#let accent = rgb("#5a2d82")

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

// ---- タイトル ----
#align(center + horizon)[
  #text(size: 30pt, weight: "bold", fill: accent)[SPR: Self-Predictive Representations]
  #v(0.6em)
  #text(size: 20pt)[Data-Efficient Reinforcement Learning with\ Self-Predictive Representations]
  #v(1.0em)
  #text(size: 17pt, fill: gray)[Schwarzer et al., ICLR 2021 ・ 論文紹介 ・ 2026-07-17]
]
#pagebreak()

#slide[一言で][
- 自己教師あり *補助損失*: 「*自分の未来の latent を当てる*」
- 報酬に頼らず、遷移の*予測可能性*から表現を学ぶ
- ベースの RL（Rainbow, off-policy）に付加 → Atari 100k で大幅改善
- 本紹介の焦点: *転移関数* ・ *EMA target* ・ *データ拡張*
]

#slide[全体像][
- online encoder: $ z_t = f_o(s_t) $
- 転移モデル $h$ で K ステップ先を*行動条件付き*で予測
- EMA target encoder $f_m$ が「正解」latent を生成
- 予測と target を projection 空間で *cosine* により一致させる
#v(0.4em)
#align(center)[#text(fill: accent)[画像 → z → （行動で）未来 z を予測 → EMA target と一致]]
]

#slide[コア (1)：転移関数][
- 現在の潜在: $ z_t = f_o(s_t) $
- 行動条件付きで再帰展開（$k = 1 dots K$, 原論文 $K = 5$）:
#align(center)[$ hat(z)_(t+k) = h( hat(z)_(t+k-1), a_(t+k-1) ), quad hat(z)_t = z_t $]
- *潜在ダイナミクス*を学ぶ = 「この行動で世界がどう変わるか」を latent 上で予測
- 原論文: 畳み込み転移（空間特徴マップ上）
- 本プロジェクト実装: MLP `transition(concat(z, a)) → z`
]

#slide[コア (2)：EMA target encoder][
- target は online の*指数移動平均*（momentum, BYOL 式）:
#align(center)[$ theta_m <- tau thin theta_m + (1 - tau) thin theta_o $]
- 正解 latent: $ tilde(z)_(t+k) = f_m(s_(t+k)) $ 、*stop-gradient*
- 非対称構造: *online 側のみ* predictor $q$（target 側にはなし）
- 役割: 表現の*崩壊 (collapse) を防ぎ*、ゆっくり動く安定した目標を与える
- 本プロジェクト実装: `target_encoder`, EMA `τ = 0.99`, `requires_grad_(False)`
]

#slide[損失][
- projection 空間での *cosine 類似度*を K ステップ和:
#align(center)[$ L_"SPR" = - sum_(k=1)^K (q(g_o(hat(z)_(t+k))) dot g_m(tilde(z)_(t+k))) / (||q(g_o(hat(z)_(t+k)))|| thin ||g_m(tilde(z)_(t+k))||) $]
- $g_o$: online projection、$g_m$: target projection（EMA）
- RL 損失に加える*補助項*（表現学習と方策学習を同時に）
]

#slide[データ拡張][
- 画像拡張: *random shift / crop + intensity*（DrQ 流）
- 系列 $(t, t+k)$ には*同一の拡張*を適用
  - → 偽の運動を注入しない（真の微小変位を潰さない）
- 拡張ありで表現がさらに*頑健*に（データ効率 benchmark で寄与大）
]

#slide[（対照）本プロジェクトでの実装][
- `src/cv_extractor/spr_state.py` = 原論文を*忠実に再現*
- encoder: ResNet18 + FPN + coord → z（128 次元）
- EMA target encoder（`τ = 0.99`）、行動条件付き MLP transition
- projector + predictor、cosine loss
- この z を*凍結*して off-policy RL（DrQ-v2）で使用 → success 100%
#v(0.3em)
#text(fill: accent)[= 「表現は自己教師ありで十分学べ、方策は別アルゴリズムで載せる」の実証]
]
