## 自行集成 / 通用 UI 开发 / AgentClient

_以下伪代码以 Python binding 为例，其他语言均有类似变动_

### 【Breaking Change】引入 add_sink 接口，废弃 notification_handler

```
# 原先
tasker = Tasker(MyNoficiationHandler())

# 现在
tasker = Tasker()
tasker.add_sink(MyTaskerEventSink())
tasker.add_context_sink(MyContextEventSink())
```

区别在于 EventSink 中额外传递了对象参数，如 tasker, context 等：
[TaskerEventSink](https://github.com/MaaXYZ/MaaFramework/blob/v5.0.0/source/binding/Python/maa/tasker.py#L538), [ContextEventSink](https://github.com/MaaXYZ/MaaFramework/blob/v5.0.0/source/binding/Python/maa/context.py#L278)

**Resource, Controller 中同样进行了该改动**

_说人话就是，把原来的 NotificationHandler 换成 EventSink，可以多拿到一个 context/tasker 参数_

老：

```
class MyNotificationHandler(NotificationHandler):
    def on_node_action(self, noti_type: NotificationType, detail: NodeActionDetail):
        pass
```

新：

```
class MyContextEventSink(ContextEventSink):
    def on_node_action(self, context: "Context", noti_type: NotificationType, detail: NodeActionDetail):
        pass
```

### 新增 `MaaAgentClientRegisterTaskerSink` 系列函数

需要调用以使 AgentServer 得以接受上述的事件消息

### 【Breaking Change】Win32Controller 构造参数更新

创建 Win32Controller 对象时，原先传入 input_method 同时设置鼠标和键盘输入方式，现在拆分成了 mouse_method 和 keyboard_method，需要分别设置

https://github.com/MaaXYZ/MaaFramework/blob/v5.0.0/source/binding/Python/maa/controller.py#L608

### 【Breaking Change】废弃 DbgController

release 产物中不再提供，仅供内部单元测试使用。若有读取图片等需求，请自行构造 CustomController 以实现

## Custom / AgentServer

_以下伪代码以 Python binding 为例，其他语言均有类似变动_

### 【Breaking Change】run_action 返回值变化

`context.run_action` 现在会返回一个 [ActionDetail](https://github.com/MaaXYZ/MaaFramework/blob/v5.0.0/source/binding/Python/maa/define.py#L711) 而不是 `NodeDetail` 了。且无论动作是否成功，只要尝试执行了动作，都会返回（而不是失败返回 None），请通过 ActionDetail.success 判断是否执行成功。

_请注意在未尝试执行时，如 entry 不存在、node disbabled 等情况下仍会返回 None_

### 【Breaking Change】run_recognition 返回值变化

`context.run_recognition` 无论识别是否命中（有没有识别到），只要尝试进行了识别，就返回 RecoResult（而不是未命中返回 None），请通过 RecoResult.hit 判断是否命中。

_请注意在未尝试识别时，如 entry 不存在、node disbabled、image 为空等情况下仍会返回 None_

### 【Breaking Change】CustomRecognition 识别结果返回的 `detail` 字段仅支持 json

同时获取到的也是 json 了

### AgenServer 可以接收事件消息

以 ContextEvent 为例，可注册以获取任意识别/动作（即使是非 Custom）开始执行、成功/失败等时机的实时回调消息，详情参考 https://github.com/MaaXYZ/MaaFramework/blob/main/source/binding/Python/maa/context.py#L350

```
@AgentServer.context_sink()
class MyContextEventSink(ContextEventSink):
    def on_node_action(self, context: "Context", noti_type: NotificationType, detail: NodeActionDetail):
        pass
```

_需要 AgentClient（通用 UI）调用 `MaaAgentClientRegisterContextSink` 系列函数后，AgentServer 才能接收到对应消息_

### 新功能 `override_image`

`context.override_image` / `resource.override_image`

用于传入模板图片

使用示例

```
context.override_image("MyImage-A", image)
context.run_task("NewTask", { {"NewTask": { "template": "MyImage-A" }} })
```

### Python 新增 `get_node_object` 接口

```
# 老，拿到的是一个 dict
node: Dict = context.get_node_data("MyNode")

# 新，拿到的是一个 dataclass
node: JPipelineData = context.get_node_object("MyNode")
```

新增接口，老接口并未废弃。拿到的数据类型参考：[JPipelineData](https://github.com/MaaXYZ/MaaFramework/blob/v5.0.0/source/binding/Python/maa/pipeline.py#L260)

### 【Breaking Change】Python 识别结果 RecognitionDetail 中 filtered_results 字段更名

原来是 `filterd_results`，错别字，改正为 `filtered_results`

### 【Breaking Change】Javascript 调整 `Rect` 类型

从对象改为数组

```
// 旧
type Rect = {
    x: number
    y: number
    width: number
    height: number
}

// 新
type Rect = [x: number, y: number, width: number, height: number]
```
