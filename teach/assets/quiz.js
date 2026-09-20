/* 可复用的检索练习组件(所有课程共用)。
 *
 * 用法:在课程 HTML 里写
 *   <div class="quiz" data-quiz='[{"q":"题干","a":["选项1","选项2"],"correct":0,"why":"解释"}]'></div>
 * 或者用 <script type="application/json" class="quiz-data"> 放同样的数组,再给 .quiz 加
 * data-quiz-src="#该script的id"。选项**必须等长**(词数与字符数尽量一致),
 * 免得学习者靠长短猜答案。
 *
 * 设计要点(为 storage strength 服务):
 * - 点了才给反馈,且立刻给 —— 反馈回路越紧越好;
 * - 答错不隐藏正确答案,但会明说错在哪(错误本身是有价值的编码时机);
 * - 每题独立作答,不做总分排名(避免把注意力从"想"转到"刷分")。
 */
(function () {
  function build(container, items) {
    container.innerHTML = "";
    items.forEach(function (item, qi) {
      var block = document.createElement("div");
      block.className = "quiz-q";

      var stem = document.createElement("p");
      stem.className = "quiz-stem";
      stem.innerHTML = "<span class='quiz-num'>" + (qi + 1) + "</span>" + item.q;
      block.appendChild(stem);

      var list = document.createElement("div");
      list.className = "quiz-opts";
      var answered = false;

      item.a.forEach(function (text, oi) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quiz-opt";
        btn.textContent = text;
        btn.addEventListener("click", function () {
          if (answered) return;
          answered = true;
          var right = oi === item.correct;
          list.querySelectorAll(".quiz-opt").forEach(function (b, bi) {
            b.disabled = true;
            if (bi === item.correct) b.classList.add("is-correct");
            else if (bi === oi) b.classList.add("is-wrong");
          });
          var fb = document.createElement("p");
          fb.className = "quiz-why " + (right ? "ok" : "no");
          fb.innerHTML = "<strong>" + (right ? "对。" : "不对。") + "</strong>" + item.why;
          block.appendChild(fb);
        });
        list.appendChild(btn);
      });

      block.appendChild(list);
      container.appendChild(block);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".quiz").forEach(function (container) {
      var raw = container.getAttribute("data-quiz");
      if (!raw) {
        var sel = container.getAttribute("data-quiz-src");
        if (sel) raw = document.querySelector(sel).textContent;
      }
      build(container, JSON.parse(raw));
    });
  });
})();
