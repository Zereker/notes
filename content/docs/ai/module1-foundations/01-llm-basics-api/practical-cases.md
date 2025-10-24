---
title: "实战案例集合"
weight: 3
---

# 实战案例集合

本节包含5个完整的实战案例，从车险反欺诈到运维事件处置，涵盖了大模型API在不同业务场景中的应用。

## 案例1：车险反欺诈（基于ChatGLM）

### 业务背景
车险欺诈是保险行业面临的重大挑战，传统规则引擎难以应对复杂多变的欺诈手段。使用大模型可以通过自然语言理解和推理能力，识别潜在的欺诈风险。

### 技术架构
```python
import requests
import json
from datetime import datetime
from typing import Dict, List, Any

class InsuranceFraudDetector:
    """车险反欺诈检测系统"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
    def _call_chatglm(self, messages: List[Dict], temperature: float = 0.3) -> str:
        """调用ChatGLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "glm-4",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API调用失败: {response.status_code}")
    
    def analyze_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析车险理赔案例"""
        
        # 构建分析提示词
        system_prompt = """你是一个专业的车险反欺诈分析师。请分析以下理赔案例，识别可能的欺诈风险点。

分析维度：
1. 时间异常：事故时间、报案时间、就医时间的合理性
2. 地点异常：事故地点与车主常住地、工作地的关系
3. 伤情异常：伤情严重程度与事故描述是否匹配
4. 费用异常：医疗费用、维修费用是否合理
5. 行为异常：车主行为、证人证言是否可疑

输出格式：
{
  "risk_level": "低风险/中风险/高风险",
  "risk_score": 0-100,
  "risk_points": ["风险点1", "风险点2"],
  "detailed_analysis": "详细分析说明",
  "recommendations": ["建议1", "建议2"]
}"""

        user_content = f"""
理赔案例信息：
- 案例编号：{claim_data.get('claim_id', 'N/A')}
- 事故时间：{claim_data.get('accident_time', 'N/A')}
- 报案时间：{claim_data.get('report_time', 'N/A')}
- 事故地点：{claim_data.get('accident_location', 'N/A')}
- 车主信息：{claim_data.get('owner_info', 'N/A')}
- 事故描述：{claim_data.get('accident_description', 'N/A')}
- 伤亡情况：{claim_data.get('injury_info', 'N/A')}
- 理赔金额：{claim_data.get('claim_amount', 'N/A')}
- 医疗费用：{claim_data.get('medical_cost', 'N/A')}
- 维修费用：{claim_data.get('repair_cost', 'N/A')}
- 其他信息：{claim_data.get('other_info', 'N/A')}

请进行全面的欺诈风险分析。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        analysis_result = self._call_chatglm(messages, temperature=0.3)
        
        try:
            # 尝试解析JSON结果
            import re
            json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                # 如果无法解析JSON，返回文本分析
                result_dict = {
                    "risk_level": "需要人工审核",
                    "risk_score": 50,
                    "risk_points": ["AI分析结果需要人工解读"],
                    "detailed_analysis": analysis_result,
                    "recommendations": ["人工复核"]
                }
        except json.JSONDecodeError:
            result_dict = {
                "risk_level": "分析异常",
                "risk_score": 0,
                "risk_points": ["JSON解析失败"],
                "detailed_analysis": analysis_result,
                "recommendations": ["重新分析"]
            }
        
        return result_dict

# 使用示例
def demo_fraud_detection():
    """演示车险反欺诈检测"""
    
    # 模拟理赔数据
    claim_case = {
        "claim_id": "CLM20241024001",
        "accident_time": "2024-10-23 02:30:00",
        "report_time": "2024-10-23 09:15:00",
        "accident_location": "北京市朝阳区某偏僻路段",
        "owner_info": "张某，男，35岁，无不良记录",
        "accident_description": "夜间行驶时与路边护栏碰撞，车辆受损严重",
        "injury_info": "驾驶员轻微擦伤，乘客无伤",
        "claim_amount": "150000元",
        "medical_cost": "5000元",
        "repair_cost": "145000元",
        "other_info": "事故现场无监控，仅有车主自述"
    }
    
    # 注意：需要真实的API Key
    # detector = InsuranceFraudDetector("your-chatglm-api-key")
    # result = detector.analyze_claim(claim_case)
    
    # 模拟分析结果
    mock_result = {
        "risk_level": "中风险",
        "risk_score": 65,
        "risk_points": [
            "事故发生在凌晨时段，且地点偏僻",
            "维修费用占理赔金额比例过高",
            "现场缺乏第三方证据",
            "报案时间延迟较长"
        ],
        "detailed_analysis": "该案例存在多个可疑点：1) 事故时间为凌晨2:30，属于事故高发时段但监管薄弱；2) 维修费用14.5万占总理赔的96.7%，比例异常；3) 现场无监控等客观证据；4) 报案延迟近7小时。建议进一步调查。",
        "recommendations": [
            "实地勘察事故现场",
            "核实维修费用明细",
            "调取车主近期行驶轨迹",
            "联系事故现场周边商户了解情况"
        ]
    }
    
    print("=== 车险反欺诈分析结果 ===")
    print(f"风险等级: {mock_result['risk_level']}")
    print(f"风险评分: {mock_result['risk_score']}/100")
    print(f"风险点: {', '.join(mock_result['risk_points'])}")
    print(f"详细分析: {mock_result['detailed_analysis']}")
    print(f"处理建议: {'; '.join(mock_result['recommendations'])}")

demo_fraud_detection()
```

## 案例2：情感分析（Qwen）

### 业务场景
电商平台需要分析用户评论的情感倾向，帮助商家了解产品满意度和改进方向。

### 实现代码
```python
import dashscope
from typing import List, Dict, Any
import re

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
    
    def analyze_single_review(self, review_text: str) -> Dict[str, Any]:
        """分析单条评论的情感"""
        
        system_prompt = """你是一个专业的情感分析专家。请分析用户评论的情感倾向。

分析要求：
1. 情感极性：正面、负面、中性
2. 情感强度：1-5分（1=非常弱，5=非常强）
3. 情感关键词：提取表达情感的关键词
4. 主要关注点：用户关心的产品特性
5. 改进建议：基于负面评论提出改进方向

输出JSON格式：
{
  "sentiment": "正面/负面/中性",
  "intensity": 1-5,
  "confidence": 0.0-1.0,
  "keywords": ["关键词1", "关键词2"],
  "aspects": ["产品特性1", "产品特性2"],
  "suggestions": ["改进建议1", "改进建议2"]
}"""

        user_content = f"请分析以下用户评论的情感：\n\n{review_text}"
        
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        
        if response.status_code == 200:
            result_text = response.output.message.content
            
            # 解析JSON结果
            try:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    import json
                    return json.loads(json_match.group())
            except:
                pass
            
            # 如果JSON解析失败，返回文本分析
            return {
                "sentiment": "未知",
                "intensity": 0,
                "confidence": 0.0,
                "keywords": [],
                "aspects": [],
                "suggestions": [],
                "raw_analysis": result_text
            }
        else:
            raise Exception(f"API调用失败: {response.code}")
    
    def batch_analyze(self, reviews: List[str]) -> Dict[str, Any]:
        """批量分析评论情感"""
        
        results = []
        sentiment_stats = {"正面": 0, "负面": 0, "中性": 0}
        all_keywords = []
        all_aspects = []
        all_suggestions = []
        
        for i, review in enumerate(reviews):
            print(f"分析评论 {i+1}/{len(reviews)}")
            
            try:
                result = self.analyze_single_review(review)
                results.append({
                    "review": review,
                    "analysis": result
                })
                
                # 统计情感分布
                sentiment = result.get("sentiment", "中性")
                sentiment_stats[sentiment] = sentiment_stats.get(sentiment, 0) + 1
                
                # 收集关键词和方面
                all_keywords.extend(result.get("keywords", []))
                all_aspects.extend(result.get("aspects", []))
                all_suggestions.extend(result.get("suggestions", []))
                
            except Exception as e:
                print(f"分析评论失败: {e}")
                results.append({
                    "review": review,
                    "analysis": {"error": str(e)}
                })
        
        # 统计词频
        from collections import Counter
        keyword_freq = Counter(all_keywords)
        aspect_freq = Counter(all_aspects)
        
        return {
            "total_reviews": len(reviews),
            "sentiment_distribution": sentiment_stats,
            "top_keywords": keyword_freq.most_common(10),
            "top_aspects": aspect_freq.most_common(10),
            "improvement_suggestions": list(set(all_suggestions)),
            "detailed_results": results
        }

# 使用示例
def demo_sentiment_analysis():
    """演示情感分析功能"""
    
    # 模拟用户评论数据
    sample_reviews = [
        "这个手机拍照效果真的很棒，电池续航也不错，物流很快，非常满意！",
        "质量一般般，价格偏高，感觉不值这个价钱，客服态度还行。",
        "产品收到了，包装很好，还没使用，看起来不错，后续使用了再来评价。",
        "用了一个星期，发现有些小问题，充电速度比较慢，但是外观设计很漂亮。",
        "非常失望！收到的产品有瑕疵，而且与描述不符，申请退货中。",
        "朋友推荐买的，确实不错，性价比很高，推荐给大家！",
        "发货速度快，包装精美，产品质量可以，总体满意，会回购的。",
        "使用一个月后来评价，稳定性不错，功能齐全，值得购买。"
    ]
    
    # 模拟分析结果（实际使用时需要真实API Key）
    mock_analysis_results = {
        "total_reviews": 8,
        "sentiment_distribution": {"正面": 5, "负面": 2, "中性": 1},
        "top_keywords": [
            ("不错", 3), ("满意", 2), ("质量", 2), ("价格", 2),
            ("快", 2), ("推荐", 2), ("包装", 2), ("外观", 1)
        ],
        "top_aspects": [
            ("产品质量", 4), ("物流配送", 3), ("价格性价比", 3),
            ("外观设计", 2), ("客户服务", 1), ("功能特性", 2)
        ],
        "improvement_suggestions": [
            "提升产品质量控制",
            "优化价格策略",
            "改进充电速度",
            "加强客服培训"
        ]
    }
    
    print("=== 用户评论情感分析报告 ===")
    print(f"总评论数: {mock_analysis_results['total_reviews']}")
    print(f"情感分布: {mock_analysis_results['sentiment_distribution']}")
    
    print("\n高频关键词:")
    for keyword, freq in mock_analysis_results['top_keywords']:
        print(f"  {keyword}: {freq}次")
    
    print("\n主要关注方面:")
    for aspect, freq in mock_analysis_results['top_aspects']:
        print(f"  {aspect}: {freq}次")
    
    print("\n改进建议:")
    for suggestion in mock_analysis_results['improvement_suggestions']:
        print(f"  • {suggestion}")

demo_sentiment_analysis()
```

## 案例3：天气Function Call（Qwen）

### 业务需求
构建一个智能天气助手，支持天气查询、穿衣建议、出行规划等功能。

### 完整实现
```python
import json
import requests
from datetime import datetime, timedelta
import dashscope

class WeatherAssistant:
    """智能天气助手"""
    
    def __init__(self, api_key: str, weather_api_key: str = None):
        dashscope.api_key = api_key
        self.weather_api_key = weather_api_key
        
        # 定义所有可用的函数
        self.functions = [
            {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如北京、上海"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_weather_forecast",
                "description": "获取未来几天的天气预报",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "days": {
                            "type": "integer",
                            "description": "预报天数，1-7天",
                            "default": 3
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_clothing_advice",
                "description": "根据天气给出穿衣建议",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "temperature": {
                            "type": "number",
                            "description": "当前温度（摄氏度）"
                        },
                        "weather_condition": {
                            "type": "string",
                            "description": "天气状况，如晴天、雨天、雪天"
                        },
                        "activity": {
                            "type": "string",
                            "description": "活动类型，如上班、运动、约会",
                            "default": "日常"
                        }
                    },
                    "required": ["temperature", "weather_condition"]
                }
            },
            {
                "name": "plan_outdoor_activity",
                "description": "根据天气规划户外活动",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "activity_type": {
                            "type": "string",
                            "description": "活动类型，如跑步、野餐、摄影",
                            "default": "户外运动"
                        },
                        "duration": {
                            "type": "integer",
                            "description": "活动持续时间（小时）",
                            "default": 2
                        }
                    },
                    "required": ["city"]
                }
            }
        ]
    
    def get_current_weather(self, city: str) -> dict:
        """获取当前天气（模拟数据）"""
        # 实际使用时应调用真实的天气API
        weather_data = {
            "北京": {"temperature": 15, "condition": "晴天", "humidity": 45, "wind": "微风"},
            "上海": {"temperature": 20, "condition": "多云", "humidity": 60, "wind": "东风3级"},
            "广州": {"temperature": 25, "condition": "小雨", "humidity": 80, "wind": "南风2级"},
            "深圳": {"temperature": 26, "condition": "晴天", "humidity": 55, "wind": "微风"},
            "杭州": {"temperature": 18, "condition": "阴天", "humidity": 65, "wind": "北风1级"}
        }
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        base_weather = weather_data.get(city, {
            "temperature": 22, "condition": "多云", "humidity": 50, "wind": "微风"
        })
        
        return {
            "city": city,
            "time": current_time,
            "temperature": base_weather["temperature"],
            "condition": base_weather["condition"],
            "humidity": base_weather["humidity"],
            "wind": base_weather["wind"],
            "air_quality": "良好"
        }
    
    def get_weather_forecast(self, city: str, days: int = 3) -> dict:
        """获取天气预报"""
        current_weather = self.get_current_weather(city)
        base_temp = current_weather["temperature"]
        
        forecast = []
        conditions = ["晴天", "多云", "小雨", "阴天"]
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            forecast.append({
                "date": date,
                "temperature_high": base_temp + (i * 2),
                "temperature_low": base_temp - 5 + i,
                "condition": conditions[i % len(conditions)],
                "rain_probability": 20 + (i * 10)
            })
        
        return {
            "city": city,
            "forecast_days": days,
            "forecast": forecast
        }
    
    def get_clothing_advice(self, temperature: float, weather_condition: str, activity: str = "日常") -> dict:
        """穿衣建议"""
        advice = {"layers": [], "accessories": [], "tips": []}
        
        # 根据温度给建议
        if temperature >= 25:
            advice["layers"] = ["短袖T恤", "薄款外套（可选）"]
            advice["tips"].append("天气较热，选择透气性好的衣物")
        elif temperature >= 20:
            advice["layers"] = ["长袖衬衫", "薄外套"]
            advice["tips"].append("温度适中，可根据体感调整")
        elif temperature >= 15:
            advice["layers"] = ["长袖上衣", "夹克或毛衣"]
            advice["tips"].append("早晚较凉，建议多穿一层")
        elif temperature >= 10:
            advice["layers"] = ["保暖内衣", "毛衣", "厚外套"]
            advice["tips"].append("天气偏冷，注意保暖")
        else:
            advice["layers"] = ["保暖内衣", "毛衣", "羽绒服"]
            advice["tips"].append("天气寒冷，需要充分保暖")
        
        # 根据天气状况调整
        if "雨" in weather_condition:
            advice["accessories"].extend(["雨伞", "防水鞋"])
            advice["tips"].append("有降雨，记得带伞")
        elif "雪" in weather_condition:
            advice["accessories"].extend(["防滑鞋", "帽子", "手套"])
            advice["tips"].append("有降雪，注意防滑保暖")
        elif "风" in weather_condition:
            advice["tips"].append("风力较大，选择贴身的衣物")
        
        # 根据活动类型调整
        if activity == "运动":
            advice["tips"].append("选择速干材质，便于运动")
        elif activity == "正式场合":
            advice["tips"].append("选择正装，注意搭配")
        
        return {
            "temperature": temperature,
            "weather": weather_condition,
            "activity": activity,
            "advice": advice
        }
    
    def plan_outdoor_activity(self, city: str, activity_type: str = "户外运动", duration: int = 2) -> dict:
        """户外活动规划"""
        weather = self.get_current_weather(city)
        forecast = self.get_weather_forecast(city, 1)
        
        recommendations = []
        best_time = []
        preparations = []
        
        # 根据天气条件给出建议
        if weather["condition"] == "晴天":
            recommendations.append("天气晴朗，非常适合户外活动")
            best_time.append("全天候适宜")
        elif weather["condition"] == "多云":
            recommendations.append("多云天气，适合户外活动，紫外线较弱")
            best_time.append("全天候适宜")
        elif "雨" in weather["condition"]:
            recommendations.append("有降雨，建议选择室内活动或延期")
            preparations.append("如果必须外出，准备好雨具")
        
        # 根据温度调整
        temp = weather["temperature"]
        if temp > 30:
            recommendations.append("温度较高，注意防暑降温")
            best_time.extend(["上午10点前", "下午4点后"])
            preparations.extend(["充足饮水", "防晒用品"])
        elif temp < 5:
            recommendations.append("温度较低，注意保暖")
            best_time.append("中午时段较为适宜")
            preparations.append("保暖装备")
        
        # 根据活动类型调整
        if activity_type == "跑步":
            preparations.extend(["运动鞋", "速干衣物", "水壶"])
        elif activity_type == "野餐":
            preparations.extend(["野餐垫", "食物保温", "垃圾袋"])
        elif activity_type == "摄影":
            preparations.extend(["相机防护", "备用电池", "三脚架"])
        
        return {
            "city": city,
            "activity_type": activity_type,
            "duration": duration,
            "current_weather": weather,
            "recommendations": recommendations,
            "best_time": best_time,
            "preparations": preparations
        }
    
    def chat(self, user_message: str) -> str:
        """智能对话接口"""
        tools = [{"type": "function", "function": func} for func in self.functions]
        
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的天气助手，可以：
1. 查询当前天气和天气预报
2. 根据天气提供穿衣建议
3. 帮助规划户外活动
4. 提供出行建议

请根据用户需求选择合适的功能，并提供专业、实用的建议。"""
            },
            {"role": "user", "content": user_message}
        ]
        
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        if response.status_code != 200:
            return f"API调用失败: {response.code}"
        
        assistant_message = response.output.message
        
        # 检查是否需要调用函数
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            messages.append(assistant_message)
            
            # 处理函数调用
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # 执行对应的函数
                if function_name == "get_current_weather":
                    result = self.get_current_weather(**function_args)
                elif function_name == "get_weather_forecast":
                    result = self.get_weather_forecast(**function_args)
                elif function_name == "get_clothing_advice":
                    result = self.get_clothing_advice(**function_args)
                elif function_name == "plan_outdoor_activity":
                    result = self.plan_outdoor_activity(**function_args)
                else:
                    result = {"error": f"未知函数: {function_name}"}
                
                # 添加函数结果
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tool_call.id
                })
            
            # 生成最终回复
            final_response = dashscope.Generation.call(
                model='qwen-plus',
                messages=messages
            )
            
            return final_response.output.message.content
        else:
            return assistant_message.content

# 使用示例
def demo_weather_assistant():
    """演示天气助手功能"""
    
    test_queries = [
        "北京今天天气怎么样？",
        "上海未来三天的天气预报",
        "今天北京15度，晴天，我要去跑步，应该穿什么？",
        "明天想在杭州拍照，天气合适吗？",
        "广州这几天适合户外野餐吗？",
        "深圳今天温度26度，多云，推荐一下穿衣搭配"
    ]
    
    # 模拟回复（实际使用需要真实API Key）
    mock_responses = [
        "北京今天天气晴朗，温度15°C，湿度45%，微风。空气质量良好，适合外出活动。",
        "上海未来三天天气：明天多云，20-15°C；后天小雨，22-17°C；大后天阴天，24-19°C。",
        "北京今天15°C晴天，适合跑步。建议穿着：长袖运动上衣+运动裤，记得携带水壶，全天候适宜运动。",
        "杭州明天阴天，18°C，湿度65%。天气条件良好，适合摄影活动。建议准备相机防护、备用电池和三脚架。",
        "广州目前小雨，温度25°C。不太适合户外野餐，建议选择室内活动或延期。如必须外出，请准备雨具。",
        "深圳26°C多云天气很舒适。推荐穿着：短袖T恤+薄款外套（可选）。天气较热，选择透气性好的衣物，全天候适宜外出。"
    ]
    
    print("=== 智能天气助手演示 ===")
    for i, query in enumerate(test_queries):
        print(f"\n用户: {query}")
        print(f"助手: {mock_responses[i]}")

demo_weather_assistant()
```

## 案例4：表格提取（Qwen-VL）

### 业务场景
财务部门需要从扫描的发票、报表图片中提取结构化数据，提高数据录入效率。

### 实现方案
```python
import base64
import json
from typing import Dict, List, Any
import dashscope

class TableExtractor:
    """表格数据提取器"""
    
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
    
    def extract_table_from_image(self, image_path: str, table_type: str = "general") -> Dict[str, Any]:
        """从图片中提取表格数据"""
        
        # 读取图片并编码
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 根据表格类型定制提示词
        prompts = {
            "invoice": """请从这张发票图片中提取所有信息，并按以下JSON格式输出：
{
  "invoice_info": {
    "invoice_number": "发票号码",
    "date": "开票日期",
    "seller": "销售方信息",
    "buyer": "购买方信息",
    "total_amount": "总金额"
  },
  "items": [
    {
      "name": "商品名称",
      "quantity": "数量",
      "unit_price": "单价",
      "amount": "金额"
    }
  ]
}""",
            "financial": """请从这张财务报表中提取数据，并按以下JSON格式输出：
{
  "report_type": "报表类型",
  "period": "报告期间",
  "data": [
    {
      "category": "科目名称",
      "current_period": "本期金额",
      "previous_period": "上期金额",
      "percentage": "百分比"
    }
  ]
}""",
            "general": """请分析这张图片中的表格，提取所有数据并按以下JSON格式输出：
{
  "table_title": "表格标题",
  "headers": ["列标题1", "列标题2", "列标题3"],
  "rows": [
    ["数据1", "数据2", "数据3"],
    ["数据4", "数据5", "数据6"]
  ],
  "summary": "表格总结"
}"""
        }
        
        prompt = prompts.get(table_type, prompts["general"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages
            )
            
            if response.status_code == 200:
                result_text = response.output.choices[0].message.content
                
                # 尝试解析JSON
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                    return {
                        "success": True,
                        "data": extracted_data,
                        "raw_text": result_text
                    }
                else:
                    return {
                        "success": False,
                        "error": "无法解析JSON格式",
                        "raw_text": result_text
                    }
            else:
                return {
                    "success": False,
                    "error": f"API调用失败: {response.code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"处理异常: {str(e)}"
            }
    
    def validate_extracted_data(self, data: Dict[str, Any], table_type: str) -> Dict[str, Any]:
        """验证提取的数据"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if table_type == "invoice":
            # 验证发票数据
            invoice_info = data.get("invoice_info", {})
            items = data.get("items", [])
            
            # 检查必填字段
            required_fields = ["invoice_number", "date", "total_amount"]
            for field in required_fields:
                if not invoice_info.get(field):
                    validation_result["errors"].append(f"缺少必填字段: {field}")
            
            # 验证金额计算
            if items:
                calculated_total = sum(
                    float(str(item.get("amount", 0)).replace(",", "").replace("¥", "").replace("元", ""))
                    for item in items
                )
                declared_total = float(str(invoice_info.get("total_amount", 0)).replace(",", "").replace("¥", "").replace("元", ""))
                
                if abs(calculated_total - declared_total) > 0.01:
                    validation_result["warnings"].append(f"金额计算不匹配: 计算总额{calculated_total}, 声明总额{declared_total}")
        
        elif table_type == "financial":
            # 验证财务报表数据
            data_rows = data.get("data", [])
            if not data_rows:
                validation_result["errors"].append("未提取到财务数据")
            
            # 检查数据格式
            for i, row in enumerate(data_rows):
                if not isinstance(row.get("current_period"), (int, float, str)):
                    validation_result["warnings"].append(f"第{i+1}行本期金额格式异常")
        
        if validation_result["errors"]:
            validation_result["is_valid"] = False
        
        return validation_result
    
    def process_batch_images(self, image_paths: List[str], table_type: str = "general") -> List[Dict[str, Any]]:
        """批量处理图片"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"处理图片 {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # 提取数据
                extraction_result = self.extract_table_from_image(image_path, table_type)
                
                if extraction_result["success"]:
                    # 验证数据
                    validation_result = self.validate_extracted_data(
                        extraction_result["data"], 
                        table_type
                    )
                    
                    results.append({
                        "image_path": image_path,
                        "extraction": extraction_result,
                        "validation": validation_result
                    })
                else:
                    results.append({
                        "image_path": image_path,
                        "extraction": extraction_result,
                        "validation": {"is_valid": False, "errors": ["提取失败"]}
                    })
                    
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "extraction": {"success": False, "error": str(e)},
                    "validation": {"is_valid": False, "errors": [str(e)]}
                })
        
        return results

# 使用示例
def demo_table_extraction():
    """演示表格提取功能"""
    
    # 模拟提取结果
    mock_invoice_data = {
        "success": True,
        "data": {
            "invoice_info": {
                "invoice_number": "INV20241024001",
                "date": "2024-10-24",
                "seller": "XX科技有限公司",
                "buyer": "YY贸易公司",
                "total_amount": "1580.00"
            },
            "items": [
                {
                    "name": "笔记本电脑",
                    "quantity": "1",
                    "unit_price": "1500.00",
                    "amount": "1500.00"
                },
                {
                    "name": "鼠标",
                    "quantity": "2",
                    "unit_price": "40.00",
                    "amount": "80.00"
                }
            ]
        }
    }
    
    mock_financial_data = {
        "success": True,
        "data": {
            "report_type": "资产负债表",
            "period": "2024年第三季度",
            "data": [
                {
                    "category": "流动资产",
                    "current_period": "5,000,000",
                    "previous_period": "4,800,000",
                    "percentage": "4.17%"
                },
                {
                    "category": "固定资产",
                    "current_period": "8,000,000",
                    "previous_period": "8,200,000",
                    "percentage": "-2.44%"
                }
            ]
        }
    }
    
    print("=== 表格数据提取演示 ===")
    
    print("\n1. 发票数据提取:")
    invoice_data = mock_invoice_data["data"]
    print(f"发票号: {invoice_data['invoice_info']['invoice_number']}")
    print(f"开票日期: {invoice_data['invoice_info']['date']}")
    print(f"总金额: ¥{invoice_data['invoice_info']['total_amount']}")
    print("商品明细:")
    for item in invoice_data["items"]:
        print(f"  - {item['name']}: {item['quantity']}个 × ¥{item['unit_price']} = ¥{item['amount']}")
    
    print("\n2. 财务报表提取:")
    financial_data = mock_financial_data["data"]
    print(f"报表类型: {financial_data['report_type']}")
    print(f"报告期间: {financial_data['period']}")
    print("数据明细:")
    for row in financial_data["data"]:
        print(f"  - {row['category']}: 本期¥{row['current_period']}, 上期¥{row['previous_period']}, 变化{row['percentage']}")

demo_table_extraction()
```

## 案例5：运维事件处置（Qwen）

### 业务场景
IT运维团队需要智能化处理各种系统故障和告警事件，提供快速的故障诊断和解决方案。

### 完整实现
```python
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import dashscope

class ITOperationsAssistant:
    """IT运维智能助手"""
    
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
        
        # 知识库：常见问题和解决方案
        self.knowledge_base = {
            "服务器": {
                "高CPU使用率": [
                    "检查占用CPU最高的进程",
                    "分析是否有异常进程或死循环",
                    "考虑增加服务器资源或优化代码",
                    "检查是否有病毒或恶意软件"
                ],
                "内存不足": [
                    "清理不必要的进程",
                    "检查内存泄漏",
                    "增加交换空间",
                    "升级物理内存"
                ],
                "磁盘空间不足": [
                    "清理临时文件和日志",
                    "删除不必要的文件",
                    "扩展磁盘容量",
                    "设置自动清理策略"
                ]
            },
            "网络": {
                "连接超时": [
                    "检查网络连接状态",
                    "验证防火墙设置",
                    "检查DNS解析",
                    "测试网络延迟和丢包率"
                ],
                "带宽不足": [
                    "分析网络流量模式",
                    "优化数据传输",
                    "升级网络带宽",
                    "实施流量控制"
                ]
            },
            "数据库": {
                "连接池耗尽": [
                    "检查数据库连接配置",
                    "优化SQL查询性能",
                    "增加连接池大小",
                    "检查是否有长时间未释放的连接"
                ],
                "查询性能慢": [
                    "分析慢查询日志",
                    "检查索引使用情况",
                    "优化SQL语句",
                    "考虑数据库分区或分库"
                ]
            }
        }
    
    def analyze_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析IT事件"""
        
        system_prompt = """你是一个资深的IT运维专家，擅长快速诊断和解决各种技术问题。
        
请分析IT事件，提供以下信息：
1. 事件级别（低/中/高/紧急）
2. 可能的根本原因
3. 紧急处理步骤
4. 详细解决方案
5. 预防措施
6. 需要的资源和时间估计

输出JSON格式：
{
  "severity": "事件级别",
  "category": "问题分类",
  "root_causes": ["可能原因1", "可能原因2"],
  "immediate_actions": ["紧急处理1", "紧急处理2"],
  "detailed_solution": ["详细步骤1", "详细步骤2"],
  "prevention": ["预防措施1", "预防措施2"],
  "resources_needed": ["所需资源1", "所需资源2"],
  "estimated_time": "预计解决时间",
  "escalation": "是否需要升级"
}"""

        user_content = f"""
事件信息：
- 事件ID: {incident_data.get('incident_id', 'N/A')}
- 发生时间: {incident_data.get('timestamp', 'N/A')}
- 系统: {incident_data.get('system', 'N/A')}
- 错误描述: {incident_data.get('description', 'N/A')}
- 错误日志: {incident_data.get('error_log', 'N/A')}
- 影响范围: {incident_data.get('impact', 'N/A')}
- 用户报告: {incident_data.get('user_report', 'N/A')}
- 系统指标: {incident_data.get('metrics', 'N/A')}

请进行全面的事件分析。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = dashscope.Generation.call(
                model='qwen-plus',
                messages=messages,
                temperature=0.3
            )
            
            if response.status_code == 200:
                result_text = response.output.message.content
                
                # 解析JSON结果
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group())
                    
                    # 补充知识库建议
                    self._enhance_with_knowledge_base(analysis_result, incident_data)
                    
                    return {
                        "success": True,
                        "analysis": analysis_result,
                        "raw_response": result_text
                    }
                else:
                    return {
                        "success": False,
                        "error": "无法解析JSON结果",
                        "raw_response": result_text
                    }
            else:
                return {
                    "success": False,
                    "error": f"API调用失败: {response.code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"分析异常: {str(e)}"
            }
    
    def _enhance_with_knowledge_base(self, analysis: Dict[str, Any], incident_data: Dict[str, Any]):
        """使用知识库增强分析结果"""
        description = incident_data.get('description', '').lower()
        
        # 匹配知识库中的解决方案
        for category, problems in self.knowledge_base.items():
            for problem, solutions in problems.items():
                if problem.lower() in description or any(keyword in description for keyword in problem.lower().split()):
                    analysis.setdefault('knowledge_base_suggestions', []).extend([
                        f"{category}-{problem}: {solution}" for solution in solutions
                    ])
    
    def generate_incident_report(self, incident_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """生成事件报告"""
        
        report_prompt = f"""
基于以下事件信息和分析结果，生成一份专业的IT事件处理报告：

事件信息：
{json.dumps(incident_data, ensure_ascii=False, indent=2)}

分析结果：
{json.dumps(analysis, ensure_ascii=False, indent=2)}

请生成包含以下部分的详细报告：
1. 事件概述
2. 影响评估
3. 根本原因分析
4. 处理过程
5. 解决方案
6. 后续行动计划
7. 经验教训

报告应该专业、简洁、易于理解。
"""

        messages = [
            {"role": "user", "content": report_prompt}
        ]
        
        try:
            response = dashscope.Generation.call(
                model='qwen-plus',
                messages=messages,
                temperature=0.5
            )
            
            if response.status_code == 200:
                return response.output.message.content
            else:
                return f"报告生成失败: {response.code}"
                
        except Exception as e:
            return f"报告生成异常: {str(e)}"
    
    def suggest_monitoring_improvements(self, incident_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于历史事件建议监控改进"""
        
        # 分析事件模式
        categories = {}
        systems = {}
        
        for incident in incident_history:
            category = incident.get('category', '未分类')
            system = incident.get('system', '未知系统')
            
            categories[category] = categories.get(category, 0) + 1
            systems[system] = systems.get(system, 0) + 1
        
        suggestions = {
            "frequent_issues": [],
            "monitoring_gaps": [],
            "automation_opportunities": [],
            "preventive_measures": []
        }
        
        # 分析高频问题
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:3]:
            suggestions["frequent_issues"].append({
                "category": category,
                "frequency": count,
                "recommendation": f"加强{category}相关的监控和预警"
            })
        
        # 监控盲点建议
        if "服务器" in categories:
            suggestions["monitoring_gaps"].append("增加服务器性能监控告警阈值")
        if "网络" in categories:
            suggestions["monitoring_gaps"].append("部署网络质量实时监控")
        if "数据库" in categories:
            suggestions["monitoring_gaps"].append("配置数据库性能监控仪表板")
        
        # 自动化机会
        suggestions["automation_opportunities"].extend([
            "自动重启服务脚本",
            "日志清理自动化",
            "健康检查自动化",
            "告警自动分级和分发"
        ])
        
        return suggestions

# 使用示例
def demo_it_operations():
    """演示IT运维助手功能"""
    
    # 模拟事件数据
    sample_incidents = [
        {
            "incident_id": "INC20241024001",
            "timestamp": "2024-10-24 14:30:00",
            "system": "Web服务器",
            "description": "服务器响应缓慢，CPU使用率持续90%以上",
            "error_log": "ERROR: High CPU usage detected, multiple processes consuming resources",
            "impact": "影响200+用户，响应时间增加5倍",
            "user_report": "网页加载非常慢，有时无法访问",
            "metrics": "CPU: 95%, Memory: 78%, Disk I/O: High"
        },
        {
            "incident_id": "INC20241024002",
            "timestamp": "2024-10-24 16:45:00",
            "system": "数据库服务器",
            "description": "数据库连接池耗尽，应用无法连接",
            "error_log": "FATAL: too many connections for role",
            "impact": "所有业务功能停止，影响全部用户",
            "user_report": "系统完全无法使用，登录失败",
            "metrics": "DB Connections: 500/500, Query Time: >30s"
        }
    ]
    
    # 模拟分析结果
    mock_analysis = {
        "severity": "高",
        "category": "性能问题",
        "root_causes": [
            "CPU密集型进程占用过多资源",
            "可能存在死循环或资源泄漏",
            "服务器配置不足以应对当前负载"
        ],
        "immediate_actions": [
            "识别并终止异常进程",
            "临时增加服务器资源",
            "启用负载均衡分散压力"
        ],
        "detailed_solution": [
            "使用top/htop命令分析CPU占用情况",
            "检查应用日志寻找异常模式",
            "优化高耗能的代码或查询",
            "考虑服务器扩容或架构调整"
        ],
        "prevention": [
            "设置CPU使用率监控告警",
            "定期进行性能测试",
            "实施代码性能审查",
            "建立容量规划流程"
        ],
        "resources_needed": [
            "运维工程师",
            "开发团队支持",
            "可能需要硬件升级"
        ],
        "estimated_time": "2-4小时",
        "escalation": "如4小时内无法解决则升级至高级工程师"
    }
    
    print("=== IT运维智能助手演示 ===")
    
    for i, incident in enumerate(sample_incidents):
        print(f"\n事件 {i+1}: {incident['incident_id']}")
        print(f"系统: {incident['system']}")
        print(f"问题: {incident['description']}")
        print(f"影响: {incident['impact']}")
        
        print(f"\n分析结果:")
        print(f"严重级别: {mock_analysis['severity']}")
        print(f"问题分类: {mock_analysis['category']}")
        print(f"预计解决时间: {mock_analysis['estimated_time']}")
        
        print(f"\n紧急处理步骤:")
        for action in mock_analysis['immediate_actions']:
            print(f"  • {action}")
        
        print(f"\n详细解决方案:")
        for solution in mock_analysis['detailed_solution']:
            print(f"  • {solution}")
        
        print("-" * 50)
    
    # 监控改进建议
    print("\n=== 监控改进建议 ===")
    monitoring_suggestions = {
        "frequent_issues": [
            {"category": "性能问题", "frequency": 8, "recommendation": "加强性能相关的监控和预警"},
            {"category": "连接问题", "frequency": 5, "recommendation": "加强连接相关的监控和预警"}
        ],
        "monitoring_gaps": [
            "增加服务器性能监控告警阈值",
            "配置数据库性能监控仪表板"
        ],
        "automation_opportunities": [
            "自动重启服务脚本",
            "日志清理自动化",
            "健康检查自动化"
        ]
    }
    
    print("高频问题:")
    for issue in monitoring_suggestions["frequent_issues"]:
        print(f"  • {issue['category']} (发生{issue['frequency']}次): {issue['recommendation']}")
    
    print("\n监控盲点:")
    for gap in monitoring_suggestions["monitoring_gaps"]:
        print(f"  • {gap}")
    
    print("\n自动化机会:")
    for opportunity in monitoring_suggestions["automation_opportunities"]:
        print(f"  • {opportunity}")

demo_it_operations()
```

## 学习总结与最佳实践

### 技术要点回顾

1. **API选择策略**
   - 根据任务类型选择合适的模型
   - 考虑成本、性能、准确性的平衡
   - 了解不同模型的特色功能

2. **Function Call设计原则**
   - 函数描述要清晰准确
   - 参数定义要完整
   - 错误处理要健壮
   - 支持多函数组合调用

3. **多模态应用技巧**
   - 图像预处理和编码
   - 提示词设计的重要性
   - 结构化数据提取方法
   - 验证和后处理机制

4. **项目实施经验**
   - 模块化设计便于维护
   - 配置化管理适应变化
   - 缓存机制优化性能
   - 监控和日志确保稳定性

### 下一步学习建议

1. **深入学习Prompt工程**
2. **掌握更多Function Call应用场景**
3. **探索RAG和Agent技术**
4. **实践更复杂的多模态应用**

通过这5个实战案例，您已经掌握了大模型API在企业级应用中的核心技能，为后续的高级应用开发奠定了坚实基础。