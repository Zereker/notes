---
title: "大模型API使用实战"
weight: 2
---

# 大模型API使用实战

## 全球AI发展现状

### 主要AI公司和模型对比

| 公司 | 主要模型 | 特点 | API价格 | 适用场景 |
|------|----------|------|---------|----------|
| **OpenAI** | GPT-4, GPT-3.5 | 综合能力强，生态成熟 | $0.01-0.06/1K tokens | 通用对话、编程、创作 |
| **Anthropic** | Claude 3 (Opus/Sonnet/Haiku) | 安全性好，长文本处理 | $0.25-15/1M tokens | 文档分析、安全场景 |
| **阿里云** | Qwen系列 | 中文优化，多模态 | ¥0.0014-0.12/1K tokens | 中文场景、多模态应用 |
| **智谱AI** | ChatGLM系列 | 开源友好，中文能力强 | ¥0.005-0.1/1K tokens | 中文对话、知识问答 |
| **百川智能** | Baichuan系列 | 中文垂直优化 | ¥0.005-0.02/1K tokens | 行业应用、中文处理 |
| **DeepSeek** | DeepSeek-V2/V3 | 性价比高，推理能力强 | $0.14-2.55/1M tokens | 代码生成、数学推理 |

### 选择模型的考虑因素

1. **任务类型**
   - 通用对话：GPT-4、Claude 3
   - 代码生成：DeepSeek、CodeLlama
   - 中文处理：Qwen、ChatGLM
   - 多模态：GPT-4V、Qwen-VL

2. **成本考虑**
   - 开发测试：选择便宜的模型
   - 生产环境：平衡性能和成本
   - 高频调用：考虑批量优惠

3. **性能要求**
   - 延迟敏感：选择推理速度快的模型
   - 准确性要求高：选择大参数模型
   - 特定领域：选择垂直优化模型

## DashScope平台使用

### 平台介绍
DashScope是阿里云推出的大模型服务平台，提供Qwen系列模型的API调用服务。

### 账号注册和API Key获取

1. **注册账号**
   ```bash
   # 访问 https://dashscope.console.aliyun.com/
   # 使用阿里云账号登录或注册新账号
   ```

2. **获取API Key**
   ```bash
   # 在控制台中创建API Key
   # 妥善保存API Key，不要泄露给他人
   ```

3. **环境配置**
   ```python
   import os
   os.environ["DASHSCOPE_API_KEY"] = "your-api-key-here"
   ```

### 基础API调用

#### 安装SDK
```bash
pip install dashscope
```

#### 基本调用示例
```python
import dashscope
from dashscope import Generation

# 设置API Key
dashscope.api_key = "your-api-key"

def simple_chat(prompt):
    """基础对话调用"""
    response = Generation.call(
        model='qwen-turbo',  # 可选：qwen-turbo, qwen-plus, qwen-max
        prompt=prompt,
        temperature=0.7,     # 控制输出随机性，0-1之间
        max_tokens=1500,     # 最大输出长度
        top_p=0.9,          # 核采样参数
    )
    
    if response.status_code == 200:
        return response.output.text
    else:
        return f"Error: {response.code} - {response.message}"

# 使用示例
result = simple_chat("请介绍一下机器学习的基本概念")
print(result)
```

#### 流式输出
```python
def stream_chat(prompt):
    """流式输出，实时显示生成内容"""
    responses = Generation.call(
        model='qwen-turbo',
        prompt=prompt,
        stream=True,  # 开启流式输出
        temperature=0.7
    )
    
    full_response = ""
    for response in responses:
        if response.status_code == 200:
            content = response.output.text
            print(content, end='', flush=True)
            full_response += content
        else:
            print(f"Error: {response.code}")
            break
    
    return full_response

# 使用示例
stream_chat("写一个Python快速排序算法")
```

#### 多轮对话
```python
def multi_turn_chat():
    """多轮对话示例"""
    messages = []
    
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break
            
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 调用API
        response = Generation.call(
            model='qwen-turbo',
            messages=messages,  # 使用消息格式而不是prompt
            temperature=0.7
        )
        
        if response.status_code == 200:
            ai_response = response.output.message.content
            print(f"AI: {ai_response}")
            
            # 添加AI回复到对话历史
            messages.append({"role": "assistant", "content": ai_response})
        else:
            print(f"Error: {response.code}")

# 运行多轮对话
multi_turn_chat()
```

### 高级功能

#### 自定义系统提示
```python
def chat_with_system_prompt(user_message, system_prompt="你是一个专业的AI助手"):
    """使用系统提示定制AI行为"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        temperature=0.7
    )
    
    return response.output.message.content

# 示例：编程助手
programming_system = """你是一个专业的Python编程助手。
- 总是提供完整、可运行的代码
- 包含详细的注释说明
- 遵循PEP 8代码规范
- 如果有多种解决方案，提供最佳实践
"""

code_result = chat_with_system_prompt(
    "写一个函数来计算列表中所有偶数的平方和",
    programming_system
)
print(code_result)
```

#### 参数调优技巧
```python
def optimized_generation(prompt, task_type="general"):
    """根据任务类型优化参数"""
    
    # 不同任务的参数配置
    configs = {
        "creative": {
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "analytical": {
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.0
        },
        "code": {
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.0
        },
        "general": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05
        }
    }
    
    config = configs.get(task_type, configs["general"])
    
    response = Generation.call(
        model='qwen-plus',
        prompt=prompt,
        **config
    )
    
    return response.output.text

# 使用示例
creative_text = optimized_generation(
    "创作一首关于秋天的诗", 
    "creative"
)

analytical_text = optimized_generation(
    "分析人工智能对就业市场的影响", 
    "analytical"
)

code_text = optimized_generation(
    "实现一个二叉搜索树的插入和查找方法", 
    "code"
)
```

## Function Call功能详解

### 什么是Function Call？

Function Call允许大模型在对话过程中调用外部函数或API，实现与外部系统的交互。

### 基本概念
```python
# Function Call的工作流程：
# 1. 用户提问 -> 2. 模型分析需要调用什么函数 -> 3. 调用函数获取数据 -> 4. 将结果返回给模型 -> 5. 模型整理后回复用户
```

### 实现天气查询Function Call

#### 1. 定义函数架构
```python
import json
import requests

# 定义天气查询函数
def get_weather(location, date=None):
    """
    获取指定地点的天气信息
    
    Args:
        location (str): 城市名称
        date (str, optional): 日期，格式YYYY-MM-DD
    
    Returns:
        dict: 天气信息
    """
    # 这里使用模拟数据，实际应该调用真实的天气API
    weather_data = {
        "北京": {"temperature": "15°C", "condition": "晴天", "humidity": "45%"},
        "上海": {"temperature": "20°C", "condition": "多云", "humidity": "60%"},
        "广州": {"temperature": "25°C", "condition": "小雨", "humidity": "80%"},
        "深圳": {"temperature": "26°C", "condition": "晴天", "humidity": "55%"}
    }
    
    result = weather_data.get(location, {
        "temperature": "未知",
        "condition": "无数据", 
        "humidity": "未知"
    })
    
    return {
        "location": location,
        "date": date or "今天",
        "weather": result
    }

# 定义函数描述，让模型知道如何使用这个函数
weather_function_schema = {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名称，例如：北京、上海、广州"
            },
            "date": {
                "type": "string",
                "description": "日期，格式YYYY-MM-DD，可选参数"
            }
        },
        "required": ["location"]
    }
}
```

#### 2. 实现Function Call处理逻辑
```python
def handle_function_call(user_message):
    """处理带Function Call的对话"""
    
    # 第一步：向模型发送消息，包含function描述
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        tools=[{
            "type": "function",
            "function": weather_function_schema
        }],
        tool_choice="auto"  # 让模型自动决定是否调用函数
    )
    
    assistant_message = response.output.message
    
    # 检查模型是否要调用函数
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        # 模型决定调用函数
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"模型决定调用函数: {function_name}")
        print(f"函数参数: {function_args}")
        
        # 执行函数调用
        if function_name == "get_weather":
            function_result = get_weather(**function_args)
        else:
            function_result = {"error": "未知函数"}
        
        # 将函数结果返回给模型
        messages.extend([
            assistant_message,  # 模型的函数调用请求
            {
                "role": "tool",
                "content": json.dumps(function_result, ensure_ascii=False),
                "tool_call_id": tool_call.id
            }
        ])
        
        # 让模型基于函数结果生成最终回复
        final_response = Generation.call(
            model='qwen-plus',
            messages=messages
        )
        
        return final_response.output.message.content
    else:
        # 模型没有调用函数，直接返回回复
        return assistant_message.content

# 使用示例
user_queries = [
    "北京今天天气怎么样？",
    "帮我查一下上海的天气",
    "广州和深圳的天气对比一下",
    "明天要去旅行，穿什么衣服合适？"  # 这个查询模型可能不会调用函数
]

for query in user_queries:
    print(f"\n用户: {query}")
    response = handle_function_call(query)
    print(f"AI: {response}")
```

#### 3. 多函数支持
```python
# 定义多个函数
def get_stock_price(symbol):
    """获取股票价格"""
    # 模拟股票数据
    stock_data = {
        "AAPL": {"price": 175.43, "change": "+2.15%"},
        "GOOGL": {"price": 2847.31, "change": "-0.82%"},
        "TSLA": {"price": 248.50, "change": "+3.27%"}
    }
    return stock_data.get(symbol, {"error": "股票代码不存在"})

def calculate_tip(bill_amount, tip_percentage=15):
    """计算小费"""
    tip = bill_amount * tip_percentage / 100
    total = bill_amount + tip
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2)
    }

# 函数描述列表
function_schemas = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "获取股票价格信息",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如AAPL、GOOGL"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate_tip",
        "description": "计算小费和总金额",
        "parameters": {
            "type": "object",
            "properties": {
                "bill_amount": {"type": "number", "description": "账单金额"},
                "tip_percentage": {"type": "number", "description": "小费百分比，默认15%"}
            },
            "required": ["bill_amount"]
        }
    }
]

# 函数映射
function_map = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "calculate_tip": calculate_tip
}

def advanced_function_call(user_message):
    """支持多函数的高级Function Call"""
    messages = [{"role": "user", "content": user_message}]
    
    tools = [{"type": "function", "function": schema} for schema in function_schemas]
    
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.output.message
    
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        # 处理函数调用
        messages.append(assistant_message)
        
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 执行对应的函数
            if function_name in function_map:
                function_result = function_map[function_name](**function_args)
            else:
                function_result = {"error": f"未知函数: {function_name}"}
            
            # 添加函数结果到消息历史
            messages.append({
                "role": "tool",
                "content": json.dumps(function_result, ensure_ascii=False),
                "tool_call_id": tool_call.id
            })
        
        # 生成最终回复
        final_response = Generation.call(
            model='qwen-plus',
            messages=messages
        )
        
        return final_response.output.message.content
    else:
        return assistant_message.content

# 测试多函数功能
test_queries = [
    "北京天气如何？",
    "AAPL股票价格是多少？",
    "账单是120元，小费按20%计算，总共多少钱？",
    "帮我查一下上海天气，还有TSLA的股价"
]

for query in test_queries:
    print(f"\n用户: {query}")
    response = advanced_function_call(query)
    print(f"AI: {response}")
```

## 多模态大模型应用

### 什么是多模态大模型？

多模态大模型能够同时处理和理解多种类型的数据，如文本、图像、音频等。

### Qwen-VL使用示例

#### 图像理解
```python
from dashscope import MultiModalConversation

def analyze_image(image_path, question):
    """分析图像并回答问题"""
    
    # 读取图像文件
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # 构建多模态消息
    messages = [
        {
            "role": "user",
            "content": [
                {"text": question},
                {"image": image_data}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    return response.output.choices[0].message.content

# 使用示例
image_analysis_questions = [
    "这张图片中有什么内容？",
    "图片中的人在做什么？",
    "这张图片的主要颜色是什么？",
    "图片中有多少个物体？"
]

# 注意：需要提供实际的图片路径
# result = analyze_image("path/to/your/image.jpg", "描述这张图片的内容")
# print(result)
```

#### 表格数据提取
```python
def extract_table_data(image_path):
    """从图像中提取表格数据"""
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": """请分析这张图片中的表格，并将数据提取为JSON格式。
                    要求：
                    1. 识别表格的行和列
                    2. 提取所有文本内容
                    3. 按照表格结构组织数据
                    4. 返回标准的JSON格式"""
                },
                {"image": image_data}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    return response.output.choices[0].message.content

# 使用示例
# table_json = extract_table_data("path/to/table_image.jpg")
# print("提取的表格数据：")
# print(table_json)
```

#### 图表分析
```python
def analyze_chart(image_path, analysis_type="general"):
    """分析图表数据"""
    
    analysis_prompts = {
        "general": "请分析这个图表，说明其类型、主要数据趋势和关键发现。",
        "trend": "分析这个图表中的数据趋势，识别上升、下降或稳定的模式。",
        "comparison": "比较图表中不同类别或时间段的数据差异。",
        "insights": "从这个图表中提取3-5个关键商业洞察。"
    }
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text": analysis_prompts.get(analysis_type, analysis_prompts["general"])},
                {"image": image_data}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    
    return response.output.choices[0].message.content

# 使用示例
# chart_analysis = analyze_chart("path/to/chart.png", "trend")
# print("图表分析结果：")
# print(chart_analysis)
```

## 错误处理和最佳实践

### 错误处理
```python
import time
from typing import Optional

def robust_api_call(prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[str]:
    """健壮的API调用，包含重试机制"""
    
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model='qwen-turbo',
                prompt=prompt,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"API错误 (尝试 {attempt + 1}/{max_retries}): {response.code} - {response.message}")
                
        except Exception as e:
            print(f"请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))  # 指数退避
    
    return None

# 使用示例
result = robust_api_call("写一个Python递归函数来计算阶乘")
if result:
    print("成功获取结果：", result)
else:
    print("API调用失败")
```

### 成本优化策略
```python
def cost_optimized_call(prompt: str, priority: str = "balanced"):
    """根据优先级选择合适的模型以优化成本"""
    
    model_configs = {
        "economy": {
            "model": "qwen-turbo",
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "balanced": {
            "model": "qwen-plus", 
            "max_tokens": 2000,
            "temperature": 0.7
        },
        "premium": {
            "model": "qwen-max",
            "max_tokens": 4000,
            "temperature": 0.8
        }
    }
    
    config = model_configs.get(priority, model_configs["balanced"])
    
    response = Generation.call(
        prompt=prompt,
        **config
    )
    
    return response.output.text

# 使用示例
economy_result = cost_optimized_call("简单问题", "economy")
premium_result = cost_optimized_call("复杂分析任务", "premium")
```

### 缓存机制
```python
import hashlib
import json
import os

class APICache:
    """简单的API响应缓存"""
    
    def __init__(self, cache_dir: str = "api_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {"prompt": prompt, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """从缓存获取结果"""
        cache_key = self._get_cache_key(prompt, **kwargs)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data.get("response")
        return None
    
    def set(self, prompt: str, response: str, **kwargs):
        """保存结果到缓存"""
        cache_key = self._get_cache_key(prompt, **kwargs)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            **kwargs
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

def cached_api_call(prompt: str, use_cache: bool = True) -> str:
    """带缓存的API调用"""
    cache = APICache()
    
    if use_cache:
        cached_result = cache.get(prompt)
        if cached_result:
            print("从缓存返回结果")
            return cached_result
    
    # 调用API
    response = Generation.call(
        model='qwen-turbo',
        prompt=prompt,
        temperature=0.7
    )
    
    if response.status_code == 200:
        result = response.output.text
        if use_cache:
            cache.set(prompt, result)
        return result
    else:
        raise Exception(f"API调用失败: {response.code}")

# 使用示例
result1 = cached_api_call("什么是深度学习？")  # 第一次调用API
result2 = cached_api_call("什么是深度学习？")  # 从缓存返回
```

通过这些实战示例，您将掌握大模型API的核心使用技巧，为后续的高级应用开发打下坚实基础。