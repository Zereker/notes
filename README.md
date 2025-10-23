# 我的学习笔记

这是我的技术学习笔记仓库，使用 Hugo + Book 主题构建。

## 在线访问

- 网站地址：https://zereker.github.io/my-learning-notes/

## 本地开发

```bash
# 克隆仓库
git clone --recursive https://github.com/Zereker/my-learning-notes.git
cd my-learning-notes

# 安装 Hugo (如果未安装)
brew install hugo

# 启动开发服务器
hugo server -D

# 构建静态文件
hugo
```

## 内容结构

```
content/
├── docs/
│   ├── programming/
│   │   └── go/           # Go 语言相关笔记
│   ├── system-design/    # 系统设计
│   ├── algorithms/       # 算法与数据结构
│   └── tools/           # 工具与技术
```

## 添加新笔记

```bash
# 创建新的笔记文件
hugo new content/docs/programming/go/new-topic.md
```

## 部署

推送到 main 分支会自动触发 GitHub Pages 部署。

## 许可证

MIT License