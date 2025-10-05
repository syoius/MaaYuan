import React, { useState, useEffect } from "react";
import {
  ChevronRight,
  ChevronLeft,
  CheckCircle,
  Circle,
  AlertCircle,
  Info,
} from "lucide-react";

const GuideApp = () => {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [completedSteps, setCompletedSteps] = useState({});
  const [expandedNotes, setExpandedNotes] = useState({});

  useEffect(() => {
    const saved = localStorage.getItem("guideProgress");
    console.log("🧩 localStorage 读取到：", saved);

    if (saved) {
      try {
        const data = JSON.parse(saved);
        console.log("✅ 恢复进度数据：", data);

        setCurrentPhase(data.phase ?? 0);
        setCompletedSteps(data.steps ?? {});
        setExpandedNotes(data.notes ?? {});
      } catch (err) {
        console.error("❌ 解析 localStorage 出错：", err);
      }
    } else {
      console.log("⚠️ 没有找到保存的进度，初始化空数据");
      const initialExpandedNotes = {};
      phases.forEach((phase) => {
        phase.steps?.forEach((step) => {
          if (step.note) initialExpandedNotes[step.id] = true;
        });
      });
      setExpandedNotes(initialExpandedNotes);
    }
  }, []);

  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("guideProgress");
    if (saved) {
      const data = JSON.parse(saved);
      setCurrentPhase(data.phase || 0);
      setCompletedSteps(data.steps || {});
      setExpandedNotes(data.notes || {});
    }
    setIsLoaded(true);
  }, []);

  useEffect(() => {
    if (!isLoaded) return;
    console.log("💾 保存进度：", {
      currentPhase,
      completedSteps,
      expandedNotes,
    });
    localStorage.setItem(
      "guideProgress",
      JSON.stringify({
        phase: currentPhase,
        steps: completedSteps,
        notes: expandedNotes,
      })
    );
  }, [isLoaded, currentPhase, completedSteps, expandedNotes]);

  // 每次切换章节时，自动滚动回页面顶端
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentPhase]);

  const phases = [
    {
      title: "跟打须知",
      description: [
        "1. 不刚需百万粮草，正常游玩即可",
        "2. 袁基线提前多存一些粮草更好，我也会把袁基线相关的资源量标注一下",
        "3. 建筑升级思路：一开始就把产量建筑和锻坊点到 5 级，之后优先点锻坊，在一队 40 级后只点粮草",
      ],
      tips: [
        "本规划适用于袁基线、左慈线、傅融线",
        "刘辩线可用钟繇替代张鲁（偷袭五斗米改为偷袭幽州）、太史慈替代张修",
        "孙策线可参考阵容思路手动打，或者期待其他自动阵容",
      ],
      achievements: [
        "✅ 不侦查",
        "✅ 不探查",
        "✅ 不协助",
        "✅ 100 回合内完成",
        "✅ 20 密探以内通关（通关后自由招人解锁20/30/50的成就）",
        "✅ 所有城池保卫成功",
        "✅ 不浪费金钲（每回合人工确认清空行动力）",
        "✅ 一局内消灭所有势力（傅融线）",
      ],
    },
    {
      title: "第四章",
      subtitle: "袁基线参考初始资源：16回合开始 戈5116 粮10772",
      goals: [
        {
          name: "全自动攻城队 - 5人",
          details: "20级荀攸 - 20级太史慈 - 20级周忠 - 17级陈登 - 18级张鲁",
          important: "⚠️ 注意卡陈登和张鲁的等级！陈登17级，张鲁18级",
        },
        {
          name: "全自动守家刷怪队 - 4人",
          details: "20级蛋 - 20级张修 - 任意慢速 - 任意慢速",
          important: "家里的 10 级密探们不用升级，直接塞进来就行",
        },
      ],
      steps: [
        {
          id: "4-1",
          title: "开局准备",
          tasks: [
            "升级：20级太史慈",
            "引进：陈登",
            "引进：荀攸",
            "升级：17级陈登",
            "升级：20级周忠",
          ],
          note: "如果突破/升级道具不够，就用 太史慈 刷 2~3 回合",
        },
        {
          id: "4-2",
          title: "攻打江州",
          optionsPosition: "before",
          // 两个并排方案
          options: [
            {
              title: "方案 A · 双人攻城",
              body: [
                {
                  type: "formation",
                  members: ["20级周忠", "17级陈登"],
                },
              ],
              note: "不浪费经验去升多余密探，但会消耗 1k 戈 + 1w 粮草复活",
            },
            {
              title: "方案 B · 三人攻城",
              body: ["20级周忠 - 17级陈登 - 20级庞羲"],
              note: ["更稳更快，战损更低。", "但庞羲之后完全用不到。"],
            },
          ],
          // 仍有通用子任务
          tasks: [
            {
              text: "一队出发后，Maa 用二队👇挂机 4 回合到攻城战，Boss 战开自动",

              formation: ["20级太史慈", "10级张仲景", "10级蛾使"],
            },
          ],
          note: "Maa 个人设置：处理叹号+只刷低等级+自动补兵+自动迎战",
        },
        {
          id: "4-3",
          title: "攻打江州后",
          tasks: [
            "笼络 张修（监狱）",
            "太史慈队 爬到 乱军营寨 6 层",
            "邀请： 蛋（壮武）",
            {
              text: "更新爬塔阵容，继续爬到12层",
              formation: ["20级蛋", "20级张修", "17级陈登"],
            },
            "邀请：张鲁（江州）",
          ],
        },
        {
          id: "4-4",
          title: "两支队伍初步成型",
          tasks: [
            {
              text: "攻打本章的敌对势力（袁基线为涿县、蓟县），先不要再去碰五斗米",
              formation: [
                "20级荀攸",
                "20级太史慈",
                "20级周忠",
                "17级陈登",
                "18级张鲁",
              ],
            },
            {
              text: "行军期间用新的二队一直刷素材（凑数的10级够用，不要升级）",
              formation: ["20级蛋", "20级张修", "慢速凑数", "慢速凑数"],
            },
          ],
          important:
            "⚠️ 进第五章前，用 张修队 偷袭孙家和袁家拿马！不拿白不拿！",
        },
      ],
    },
    {
      title: "第五章",
      subtitle: "获得全自动攻城队完全体",
      goals: [
        {
          name: "攻城队完全体",
          details: "30级荀攸 - 30级钟繇 - 27级鸡 - 30级张鲁 - 20~28级孙权",
          important: "非袁基线可卡29级钟繇，否则董奉需骑加速马",
        },
      ],
      steps: [
        {
          id: "5-1",
          title: "开局准备",
          tasks: ["升级：30级蛋", "升级：30级张修"],
        },
        {
          id: "5-2",
          title: "攻打宛陵",
          tasks: [
            "上一章用的张鲁攻城队，不用升级，直接攻打宛陵",
            "30级张修队刷五斗米教坛到12层",
            "邀请：鸡（宛陵）",
          ],
        },
        {
          id: "5-3",
          title: "刷素材升级攻城队",
          tasks: [
            "刷 4-5 回合的素材（Maa 设置全都刷）",
            "笼络：钟繇（监狱）",
            "升级：30级荀攸",
            "升级：30级钟繇",
            "升级：30级蛋",
            "升级：30级张鲁",
            "升级：27级陈登",
            "升级：27级鸡",
          ],
          note: "都拉到30级也可以，27级是省资源的拉法",
        },
        {
          id: "5-4",
          title: "攻打吴县",
          tasks: [
            {
              text: "临时组一下吴县特攻队",
              formation: [
                "30级荀攸",
                "30级钟繇",
                "27级鸡",
                "27级陈登",
                "30级蛋",
                "30级张鲁",
              ],
            },
            "出发前检查：阵型中，蛋站在比张鲁更靠左的列（让蛋先动）",
          ],
          note: "行军 1 回合就能到，派兵前只需确认地图无倒计时 1 回合的敌军",
        },
        {
          id: "5-5",
          title: "获取孙权",
          tasks: [
            "爬宗贼营地 12 层。懒得换队就全程用吴县特攻队，想效率就前 11 层用经典 4 人张修队，第 12 层用吴县特攻队",
            "邀请：孙权（吴县）",
          ],
        },
        {
          id: "5-6",
          title: "最终编队",
          optionsPosition: "after",
          options: [
            {
              title: "袁基线 · 至此进入第六章",
            },
            {
              title: "非袁基线 · 获取董奉",
              body: [
                "攻城队打邺县",
                "张修队爬乱军营寨21层",
                "邀请：董奉（邺县）",
                "更新队伍：用30级董奉替换荀攸",
                "用董奉版攻城队清理本章敌对势力",
              ],
              note: "关于不侦查的汝阳攻城战：不太记得带董奉的30级攻城队打汝阳战损是多少了，资源紧张的话可以刷到40级再打，反正就算打完进了下一章也是一样要先刷到40级。百万粮草就随意了。",
            },
          ],
          tasks: [
            {
              text: "攻城队完全体",
              formation: [
                "30级荀攸",
                "30级钟繇",
                "27级陈登",
                "30级蛋",
                "30级张鲁",
                "20~28级孙权",
              ],
            },
            {
              text: "鸡回到二队",
              formation: ["27级鸡", "30级张修", "慢速凑数", "慢速凑数"],
            },
            "编队窍门：① 把枪卒（钟繇）摆在第一排最左边克制骑兵 ② 保持蛋在张鲁的左侧",
          ],
        },
      ],
    },
    {
      title: "第六章",
      subtitle: "刷满40级，准备连续攻打西凉+不明的 6 座城",
      goals: [
        {
          name: "袁基线攻城队",
          details:
            "40级荀攸-40级钟繇-36级陈登-40级蛋-40级张鲁-37级孙权（不可更高）",
        },
        {
          name: "非袁基线攻城队",
          details:
            "40级董奉-39级钟繇-36级陈登-40级蛋-40级张鲁-37级孙权（不可更高）",
          important: "或者40级钟繇+速度马董奉",
        },
        {
          name: "守家队",
          details: "34~40级鸡-40级张修-慢速凑数-慢速凑数",
          important:
            "非袁基线时后两位可以放荀彧郭嘉，不需要满级，得让他们速度比张修慢",
        },
      ],
      steps: [
        {
          id: "6-1",
          title: "开局准备",
          tasks: [
            "（袁基线）攻打南郑，凑够 4000 威势解锁等级上限 40 级",
            "升级：40级张修、鸡（30级张修队也能开刷）",
          ],
        },
        {
          id: "6-2",
          title:
            "刷素材升满 40 级攻城队（非袁基线可能上一章就刷满了，直接下一步）",
          tasks: [
            "（袁基线）Maa 全都刷 15 回合（我是每次挂 5 回合，这样能够比较及时去升级建筑）",
            "如果本章开局没突破张修和鸡，优先养他们",
            "升级：40级荀攸/董奉",
            "升级：39~40级钟繇",
            "升级：40级蛋",
            "升级：40级张鲁",
            "升级：36级陈登、37级孙权",
            "升级：37级孙权",
          ],
          note: [
            "Maa 个人设置：不处理叹号+全都刷+自动补兵+自动迎击。",
            "非袁基线可捞郭嘉（监狱）和荀彧（汝阳），懒就不整",
          ],
        },
        {
          id: "6-3",
          title: "攻城队出击（现在开始到通关需要22回合）",
          optionsPosition: "before",
          options: [
            {
              title: "袁基线攻城队",
              body: [
                {
                  type: "formation",
                  members: [
                    "40级荀攸",
                    "40级钟繇",
                    "36级陈登",
                    "40级蛋",
                    "40级张鲁",
                    "37级孙权",
                  ],
                  note: "这不是阵型，只是自动出手顺序。摆阵型时需要荀攸站第一排最左，蛋站张鲁左侧。",
                },
              ],
            },
            {
              title: "非袁基线攻城队",
              body: [
                {
                  type: "formation",
                  members: [
                    "40级董奉",
                    "39级钟繇",
                    "36级陈登",
                    "40级蛋",
                    "40级张鲁",
                    "37级孙权",
                  ],
                  note: "这不是阵型，只是自动出手顺序。摆阵型时需要蛋站张鲁左侧。40级钟繇需要董奉骑加速马。",
                },
              ],
            },
          ],
          tasks: [
            "攻打冀县",
            "攻打金城",
            "如果有意正面对抗下一章的 1w6 偷家贼，行军途中继续刷素材再升级一个速度大于陈登的人（可以骑马），如果不想招募新人可以直接给蛾使之类的元老升满级骑马",
          ],
          important:
            "这个队伍已经可以auto到通关！没董奉就是硬烧粮草，董奉版几乎无伤。听说有个邪修是现在直接宣战龙耆城及其他不明势力城池。这六章的不明势力不会派出偷袭兵。可以直接把龙耆城打了，之后侦查助战啥都能用了。",
        },
        {
          id: "6-4",
          title: "攻打姑臧（吕布）",
          tasks: [
            {
              text: "手动打法的编队站位：孙权-前排最左，鸡蛋后排中右",
              formation: ["40级鸡", "40级蛋", "37级孙权"],
            },
            "杂兵战：第一回合-全A盾， 第二回合-鸡蛋合兵全A骑， 第三回合-鸡蛋大",
            "李傕郭汜：第一回合-全A李， 第二回合-合兵全A郭， 第三回合鸡蛋大",
            "吕布：第一回合-鸡蛋待机 权A， 第二回合-合兵都A 第三回合-孙权死无所谓 鸡蛋大 第四回合-鸡蛋A",
          ],
          note: "有足够粮草也可以用攻城队直接auto，我只在有董奉的时候试过，前两战都ok，吕布要复活一次",
        },
      ],
    },
    {
      title: "第七章",
      subtitle: "一路攻城到通关",
      description: [
        "粮够的话无缝攻城到厌倦。无董奉每战需 3w 粮，有董奉打龙耆城 3 战总共不到 3w 粮。",
      ],
      goals: [
        {
          name: "如果想正面应对 1w6 偷家，拉扯一个人替代钟繇",
          details:
            "40级荀攸/董奉-速度大于陈登的-36级陈登-40级蛋-40级张鲁-37级孙权（不可更高）",
        },
        {
          name: "守家队",
          details: "34~40级鸡-40级张修-慢速凑数-慢速凑数",
          important:
            "遇到 1w6 偷家的把张修换成钟繇，出战两次即可。非袁基线时后两位可以放荀彧郭嘉，不需要满级，得让他们速度比张修慢。如果是荀彧郭嘉版，则对抗 1w6 时可以把荀攸张修也都塞进来组成满编队，自动没试过，手动能直接打赢。",
        },
      ],
      optionsPosition: "before",
      options: [
        {
          title: "正常推进",
          body: ["攻打龙耆城"],
        },
      ],
      steps: [
        {
          id: "7-1",
          title: "正常推进",
          tasks: ["继续使用攻城队一路攻城", "二队守家应对偷袭"],
        },
        {
          id: "7-2",
          title: "邪修方案（拿成就用）",
          tasks: [
            "直接攻打龙耆城，即使没董奉也只要6w粮草就能auto完",
            "成就到手后可以傩回宣战前",
            "花3k戈戟+3w粮草讨好不明势力",
            "关系变中立就不会派偷袭队了",
          ],
          important: "这是最快拿成就的方法！",
          achievements: ["111"],
        },
      ],
    },
  ];

  const toggleStep = (stepId) => {
    setCompletedSteps((prev) => ({
      ...prev,
      [stepId]: !prev[stepId],
    }));
  };

  const toggleTask = (stepId, taskIdx) => {
    const taskId = `${stepId}-task-${taskIdx}`;
    setCompletedSteps((prev) => ({
      ...prev,
      [taskId]: !prev[taskId],
    }));
  };

  const toggleNote = (stepId) => {
    setExpandedNotes((prev) => ({
      ...prev,
      [stepId]: !prev[stepId],
    }));
  };

  const selectOption = (stepId, optCount, idx) => {
    setCompletedSteps((prev) => {
      const next = { ...prev };
      const currentKey = `${stepId}-opt-${idx}`;
      const isCurrentlySelected = !!next[currentKey];

      // 先清空该步骤下的所有方案
      for (let i = 0; i < optCount; i++) {
        next[`${stepId}-opt-${i}`] = false;
      }

      // 如果之前已选中 -> 取消选择（不再设 true）
      // 如果之前未选中 -> 选中当前项
      if (!isCurrentlySelected) {
        next[currentKey] = true;
      }

      return next;
    });
  };
  const renderOptions = (step) => (
    <div>
      <div className="text-sm text-gray-600 mb-2">
        从以下方案中{" "}
        <span className="font-semibold text-amber-700">任选其一</span>：
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {step.options.map((opt, oIdx) => {
          const chosen = !!completedSteps[`${step.id}-opt-${oIdx}`];

          // 计算该方案的 body 是否全部完成
          const allBodyDone =
            opt.body?.length &&
            opt.body.every(
              (_, li) => completedSteps[`${step.id}-opt-${oIdx}-body-${li}`]
            );

          return (
            <div
              key={oIdx}
              className={`rounded-lg border p-3 transition-all duration-300 ${
                chosen
                  ? "border-amber-400 bg-gradient-to-br from-slate-700 to-slate-800 text-white shadow"
                  : allBodyDone
                  ? "border-green-400 bg-green-50"
                  : "border-gray-200 bg-white"
              }`}
            >
              {/* 面板标题 + 选中按钮 */}
              <div className="flex items-center justify-between mb-2">
                <div className="font-semibold text-base">{opt.title}</div>
                <button
                  onClick={() =>
                    setCompletedSteps((prev) => {
                      const next = { ...prev };
                      const key = `${step.id}-opt-${oIdx}`;
                      const wasSelected = !!next[key];
                      // 先清空该 step 下所有方案
                      for (let i = 0; i < step.options.length; i++) {
                        next[`${step.id}-opt-${i}`] = false;
                      }
                      // 再按需切换当前方案
                      if (!wasSelected) next[key] = true;
                      return next;
                    })
                  }
                  title={chosen ? "取消选择" : "选择该方案"}
                >
                  {chosen ? (
                    <CheckCircle className="w-5 h-5 text-amber-400" />
                  ) : (
                    <Circle className="w-5 h-5 text-gray-400 hover:text-amber-500" />
                  )}
                </button>
              </div>

              {/* 主体内容：body 可勾选 */}
              {opt.body && (
                <div className="space-y-2 text-sm">
                  {opt.body.map((line, li) => {
                    const bodyId = `${step.id}-opt-${oIdx}-body-${li}`;
                    const isDone = completedSteps[bodyId];

                    // formation 对象
                    if (typeof line === "object" && line.type === "formation") {
                      return (
                        <div key={li} className="mt-2">
                          {renderFormation(line)}
                        </div>
                      );
                    }

                    // formation 字符串
                    if (isFormation(line)) {
                      return (
                        <div key={li} className="mt-2">
                          {renderFormation(line)}
                        </div>
                      );
                    }

                    // 普通文本任务（可勾选）
                    return (
                      <div key={li} className="flex items-start gap-2 pl-2">
                        <button
                          onClick={() =>
                            setCompletedSteps((prev) => ({
                              ...prev,
                              [bodyId]: !prev[bodyId],
                            }))
                          }
                          className="flex-shrink-0 mt-0.5"
                        >
                          {isDone ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <Circle className="w-4 h-4 text-gray-400 hover:text-amber-500" />
                          )}
                        </button>
                        <span
                          className={`leading-relaxed ${
                            isDone
                              ? "line-through text-gray-400"
                              : chosen
                              ? "text-gray-100"
                              : "text-gray-800"
                          }`}
                        >
                          {line}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* 附注 note，选中与否样式不同 */}
              {opt.note && (
                <div
                  className={`mt-3 text-xs p-2 rounded border-l-4 transition-colors duration-300 ease-in-out ${
                    chosen
                      ? "text-blue-100 bg-slate-800/30 border-blue-400"
                      : "text-blue-900 bg-blue-50 border-blue-400"
                  }`}
                >
                  {opt.note}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  const isStepCompleted = (step) => {
    // 子任务（tasks）是否全部完成
    const tasksOk = step.tasks
      ? step.tasks.every((_, idx) => completedSteps[`${step.id}-task-${idx}`])
      : completedSteps[step.id];

    // 若有方案（options），任一方案被选中即可；否则视为通过
    const optionOk = step.options
      ? step.options.some((_, idx) => completedSteps[`${step.id}-opt-${idx}`])
      : true;

    return tasksOk && optionOk;
  };

  const renderFormation = (formationInput, compact = false) => {
    // 统一抽取成员与可选备注
    let members = [];
    let note = null;

    if (Array.isArray(formationInput)) {
      members = formationInput;
    } else if (typeof formationInput === "string") {
      members = formationInput
        .split("-")
        .map((s) => s.trim())
        .filter(Boolean);
    } else if (formationInput && typeof formationInput === "object") {
      if (Array.isArray(formationInput.members)) {
        members = formationInput.members;
      } else if (typeof formationInput.text === "string") {
        members = formationInput.text
          .split("-")
          .map((s) => s.trim())
          .filter(Boolean);
      }
      if (formationInput.note) note = formationInput.note;
    }

    if (!members.length) return null; // 没有成员就不渲染

    return (
      <div
        className={`mt-2 rounded-lg p-4 shadow-inner transition-all ${
          compact
            ? "bg-slate-100"
            : "bg-gradient-to-r from-slate-700 to-slate-800"
        }`}
      >
        <div
          className={`flex items-center gap-2 ${
            compact
              ? "text-slate-600 text-xs font-semibold mb-2"
              : "text-amber-300 text-xs font-semibold mb-3"
          }`}
        >
          <span>⚔️ 队伍配置</span>
        </div>

        <div className="flex flex-wrap gap-2">
          {members.map((m, i) => (
            <div key={i} className="flex items-center gap-2">
              <span
                className={`font-bold text-xs ${
                  compact ? "text-slate-500" : "text-amber-400"
                }`}
              >
                {i + 1}
              </span>
              <span
                className={`px-3 py-1.5 rounded-md shadow-sm text-sm ${
                  compact
                    ? "bg-slate-200 text-slate-700"
                    : "bg-slate-600 text-white"
                }`}
              >
                {m}
              </span>
            </div>
          ))}
        </div>

        {note && (
          <div
            className={`mt-2 text-xs ${
              compact
                ? "text-slate-700 bg-slate-50 border-l-4 border-slate-300"
                : "text-blue-100 bg-slate-800/30 border-l-4 border-blue-400"
            } p-2 rounded`}
          >
            {note}
          </div>
        )}
      </div>
    );
  };

  const isFormation = (text) => {
    return text.includes("级") && text.split("-").length >= 3;
  };

  const phase = phases[currentPhase];
  const progress = phase.steps
    ? (phase.steps.filter((step) => isStepCompleted(step)).length /
        phase.steps.length) *
      100
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 to-orange-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* 头部 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-amber-900 mb-2">
            Boss 战也要全部 auto 的通关规划
          </h1>
          <p className="text-gray-600 mb-4">
            进度会自动保存，关闭浏览器也不会丢失哦
          </p>

          {/* 进度条 */}
          <div className="flex items-center gap-2 mb-4">
            {phases.map((p, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentPhase(idx)}
                className="flex-1 group cursor-pointer"
              >
                <div
                  className={`h-2 rounded-full transition-all ${
                    idx < currentPhase
                      ? "bg-green-500"
                      : idx === currentPhase
                      ? "bg-amber-500"
                      : "bg-gray-200 group-hover:bg-gray-300"
                  }`}
                />
                <div
                  className={`text-xs mt-1 text-center transition-all ${
                    idx === currentPhase
                      ? "font-bold text-amber-900"
                      : "text-gray-500 group-hover:text-gray-700"
                  }`}
                >
                  {idx === 0 ? "说明" : `第${idx + 3}章`}
                </div>
                {/* 小圆点指示器 - 固定高度确保对齐 */}
                <div className="flex justify-center items-center mt-1 h-4">
                  {idx < currentPhase ? (
                    <CheckCircle className="w-3 h-3 text-green-500" />
                  ) : idx === currentPhase ? (
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                  ) : (
                    <div className="w-2 h-2 bg-gray-300 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
        {/* 当前阶段 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-amber-900 mb-1">
                {phase.title}
              </h2>
              {phase.subtitle && (
                <p className="text-gray-600">{phase.subtitle}</p>
              )}
            </div>
            {phase.steps && (
              <div className="text-right">
                <div className="text-2xl font-bold text-amber-600">
                  {Math.round(progress)}%
                </div>
                <div className="text-xs text-gray-500">完成度</div>
              </div>
            )}
          </div>

          {phase.description && (
            <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4 space-y-1">
              {phase.description.map((line, idx) => (
                <p key={idx} className="text-blue-900">
                  {line}
                </p>
              ))}
            </div>
          )}

          {/* 提示信息 */}
          {phase.tips && (
            <div className="bg-amber-50 rounded-lg p-4 mb-4">
              <h3 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
                <Info className="w-5 h-5" />
                分线相关
              </h3>
              <ul className="space-y-1">
                {phase.tips.map((tip, idx) => (
                  <li key={idx} className="text-sm text-amber-800">
                    • {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 章节目标 */}
          {phase.goals && (
            <div className="mb-6">
              <h3 className="font-bold text-gray-900 mb-3">📋 章节目标</h3>
              <div className="space-y-3">
                {phase.goals.map((goal, idx) => (
                  <div key={idx} className="bg-purple-50 rounded-lg p-4">
                    <div className="font-semibold text-purple-900 mb-1">
                      {goal.name}
                    </div>
                    {renderFormation(goal.details)}
                    {goal.important && (
                      <div className="text-sm text-red-600 mt-2 font-semibold">
                        {goal.important}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 步骤列表 */}
          {phase.steps && (
            <div className="space-y-4">
              <h3 className="font-bold text-gray-900">📝 详细步骤</h3>
              {phase.steps.map((step, idx) => (
                <div
                  key={step.id}
                  className={`border-2 rounded-lg p-4 transition-all ${
                    isStepCompleted(step)
                      ? "border-green-300 bg-green-50"
                      : "border-gray-200 bg-white"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <button
                      onClick={() => {
                        // 点击大步骤时，切换所有子任务的状态
                        const allCompleted = isStepCompleted(step);
                        step.tasks.forEach((_, taskIdx) => {
                          const taskId = `${step.id}-task-${taskIdx}`;
                          setCompletedSteps((prev) => ({
                            ...prev,
                            [taskId]: !allCompleted,
                          }));
                        });
                      }}
                      className="mt-1 flex-shrink-0"
                    >
                      {isStepCompleted(step) ? (
                        <CheckCircle className="w-6 h-6 text-green-600" />
                      ) : (
                        <Circle className="w-6 h-6 text-gray-400" />
                      )}
                    </button>

                    <div className="flex-1">
                      <h4 className="font-bold text-gray-900 mb-2">
                        {idx + 1}. {step.title}
                      </h4>
                      {/* 方案在任务前 */}
                      {step.options && step.optionsPosition !== "after" && (
                        <div className="mb-4">{renderOptions(step)}</div>
                      )}

                      <ul className="space-y-2">
                        {step.tasks.map((task, taskIdx) => {
                          const taskId = `${step.id}-task-${taskIdx}`;
                          const isTaskCompleted = completedSteps[taskId];

                          // 支持三种格式：
                          // 1️⃣ 纯字符串
                          // 2️⃣ 含 formation 属性的对象
                          // 3️⃣ 旧写法：字符串中包含队伍信息（"20级荀攸 - 20级张鲁"）
                          const taskText =
                            typeof task === "string" ? task : task.text;
                          const formationData =
                            typeof task === "object" && task.formation
                              ? task.formation
                              : isFormation(taskText)
                              ? taskText
                              : null;

                          return (
                            <li
                              key={taskIdx}
                              className="text-sm text-gray-700 space-y-2"
                            >
                              {/* ✅ 主任务文本 + 勾选按钮 */}
                              <div className="flex items-start gap-2">
                                <button
                                  onClick={() => toggleTask(step.id, taskIdx)}
                                  className="flex-shrink-0 mt-0.5"
                                >
                                  {isTaskCompleted ? (
                                    <CheckCircle className="w-4 h-4 text-green-600" />
                                  ) : (
                                    <Circle className="w-4 h-4 text-gray-400 hover:text-amber-500" />
                                  )}
                                </button>
                                <span
                                  className={
                                    isTaskCompleted
                                      ? "line-through text-gray-500"
                                      : ""
                                  }
                                  dangerouslySetInnerHTML={{ __html: taskText }}
                                ></span>
                              </div>

                              {/* ✅ 若该任务有 formation 属性或检测到阵容字符串，则渲染 formation 面板 */}
                              {formationData && (
                                <div className="ml-6">
                                  {renderFormation(formationData)}
                                </div>
                              )}
                            </li>
                          );
                        })}
                      </ul>

                      {/* 方案在任务后 */}
                      {step.options && step.optionsPosition === "after" && (
                        <div className="mt-4">{renderOptions(step)}</div>
                      )}

                      {step.note && (
                        <button
                          onClick={() => toggleNote(step.id)}
                          className="mt-3 text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
                        >
                          <AlertCircle className="w-4 h-4" />
                          {expandedNotes[step.id] ? "收起" : "查看"}注意事项
                        </button>
                      )}

                      {expandedNotes[step.id] && step.note && (
                        <div className="mt-2 bg-blue-50 border-l-4 border-blue-400 p-3 text-sm text-blue-900">
                          {step.note}
                        </div>
                      )}

                      {step.important && (
                        <div className="mt-3 bg-red-50 border-l-4 border-red-400 p-3 text-sm text-red-900 font-semibold">
                          {step.important}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* 成就列表 */}
          {phase.achievements && (
            <div className="mt-6 bg-green-50 rounded-lg p-4">
              <h3 className="font-bold text-green-900 mb-3">🏆 可达成成就</h3>
              <div className="grid grid-cols-2 gap-2">
                {phase.achievements.map((achievement, idx) => (
                  <div key={idx} className="text-sm text-green-800">
                    {achievement}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 导航按钮 */}
        <div className="flex justify-between items-center">
          <button
            onClick={() => setCurrentPhase(Math.max(0, currentPhase - 1))}
            disabled={currentPhase === 0}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              currentPhase === 0
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "bg-amber-600 text-white hover:bg-amber-700 shadow-lg"
            }`}
          >
            <ChevronLeft className="w-5 h-5" />
            上一阶段
          </button>

          <button
            onClick={() => {
              localStorage.clear();
              setCurrentPhase(0);
              setCompletedSteps({});

              // 重置后默认展开所有 notes
              const initialExpandedNotes = {};
              phases.forEach((phase) => {
                if (phase.steps) {
                  phase.steps.forEach((step) => {
                    if (step.note) {
                      initialExpandedNotes[step.id] = true;
                    }
                  });
                }
              });
              setExpandedNotes(initialExpandedNotes);
            }}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 underline"
          >
            重置进度
          </button>

          <button
            onClick={() =>
              setCurrentPhase(Math.min(phases.length - 1, currentPhase + 1))
            }
            disabled={currentPhase === phases.length - 1}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              currentPhase === phases.length - 1
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "bg-amber-600 text-white hover:bg-amber-700 shadow-lg"
            }`}
          >
            下一阶段
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        {/* 底部说明 */}
        <div className="mt-6 text-center text-sm text-gray-500"></div>
      </div>
    </div>
  );
};

export default GuideApp;
