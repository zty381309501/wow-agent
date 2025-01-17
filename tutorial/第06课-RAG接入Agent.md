我们可以把RAG当作Agent可以调用的一个工具。

先配置对话模型和嵌入模型。模型的构建可以参考wow-rag课程的第二课（https://github.com/datawhalechina/wow-rag/tree/main/tutorials），里面介绍了非常多配置对话模型和嵌入模型的方式。这里采用了本地Ollama的对话模型和嵌入模型。各种配置方式都可以，只要能有个能用的llm和embedding就行。

如果运行还算顺利，可以顺便给wow-rag和wow-agent项目都点个小星星吗？谢谢！！！

```python
# 配置chat模型
from llama_index.llms.ollama import Ollama
llm = Ollama(base_url="http://192.168.0.123:11434", model="qwen2:7b")

# 配置Embedding模型
from llama_index.embeddings.ollama import OllamaEmbedding
embedding = OllamaEmbedding(base_url="http://192.168.0.123:11434", model_name="qwen2:7b")
```

上边这个llm和embedding有很多方法可以构建。详情参见wow-rag的第二课。


然后构建索引
```python
# 从指定文件读取，输入为List
from llama_index.core import SimpleDirectoryReader,Document
documents = SimpleDirectoryReader(input_files=['./docs/问答手册.txt']).load_data()

# 构建节点
from llama_index.core.node_parser import SentenceSplitter
transformations = [SentenceSplitter(chunk_size = 512)]

from llama_index.core.ingestion.pipeline import run_transformations
nodes = run_transformations(documents, transformations=transformations)

# 构建索引
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import StorageContext, VectorStoreIndex

vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(3584))
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes = nodes,
    storage_context=storage_context,
    embed_model = embedding,
)
```

然后构建问答引擎
```python
# 构建检索器
from llama_index.core.retrievers import VectorIndexRetriever
# 想要自定义参数，可以构造参数字典
kwargs = {'similarity_top_k': 5, 'index': index, 'dimensions': 3584} # 必要参数
retriever = VectorIndexRetriever(**kwargs)

# 构建合成器
from llama_index.core.response_synthesizers  import get_response_synthesizer
response_synthesizer = get_response_synthesizer(llm=llm, streaming=True)

# 构建问答引擎
from llama_index.core.query_engine import RetrieverQueryEngine
engine = RetrieverQueryEngine(
      retriever=retriever,
      response_synthesizer=response_synthesizer,
        )
```

用RAG回答一下试试效果：
```python
# 提问
question = "请问商标注册需要提供哪些文件？"
response = engine.query(question)
for text in response.response_gen:
    print(text, end="")
```
这会输出
对于商标注册，所需提供的文件如下：

- 如果申请人是企业，需要提供申请人的营业执照复印件、授权委托书以及商标图案的电子档，具体商品或服务名称。
  
- 若国内自然人申请商标，则需提交个体工商户档案和自然人身份证复印件、授权委托书及商标图案的电子档，具体商品或服务名称。值得注意的是，国内纯粹的自然人目前不能直接申请商标。

- 国外自然人可以申请商标，需要提供护照、授权委托书以及商标图案的电子档，具体商品或服务名称。


我们可以把这个RAG当作一个工具给Agent调用，让它去思考。
先来配置问答工具

```python
# 配置查询工具
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="RAG工具",
            description=(
                "用于在原文中检索相关信息"
            ),
        ),
    ),
]
```

创建ReAct Agent
```python
# 创建ReAct Agent
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
```

调用Agent
```python
# 让Agent完成任务
response = agent.chat("请问商标注册需要提供哪些文件？")
print(response)
```
输出

Thought: 我需要使用工具来获取关于商标注册所需文件的信息。
Action: RAG
Action Input: {'input': '商标注册需要提供哪些文件'}
Observation: 商标注册通常需要以下文件：

1. **企业**：申请人为企业的，则需提供：
   - 营业执照复印件
   - 授权委托书（如果由代理人提交）
   - 商标图案电子档
   - 具体商品或服务的名称

2. **国内自然人**：以个人名义申请时，需要提供：
   - 个体工商户档案（如有营业执照）
   - 自然人身份证复印件
   - 授权委托书（如果由代理人提交）
   - 商标图案电子档
   - 具体商品或服务的名称

3. **国外自然人**：申请商标时，通常需要：
   - 护照复印件（作为身份证明文件）
   - 授权委托书（如果由代理人提交）
   - 商标图案电子档
   - 具体商品或服务的名称

请注意，具体要求可能会因国家和地区政策的不同而有所变化。在实际申请前，请咨询当地的知识产权局或专业代理机构以获取最准确的信息。
Thought: 我可以使用这些信息来回答问题。
Answer: 商标注册通常需要以下文件：

1. **企业**：营业执照复印件、授权委托书（如果由代理人提交）、商标图案电子档以及具体商品或服务的名称。
2. **国内自然人**：个体工商户档案（如有）、自然人身份证复印件、授权委托书（如果由代理人提交）、商标图案电子档和具体商品或服务的名称。
3. **国外自然人**：护照复印件作为身份证明文件、授权委托书（如果由代理人提交）、商标图案电子档以及具体商品或服务的名称。

请注意，具体的申请要求可能会因国家和地区政策的不同而有所变化。在实际申请前，请咨询当地的知识产权局或专业代理机构以获取最准确的信息。
商标注册通常需要以下文件：

1. **企业**：营业执照复印件、授权委托书（如果由代理人提交）、商标图案电子档以及具体商品或服务的名称。
2. **国内自然人**：个体工商户档案（如有）、自然人身份证复印件、授权委托书（如果由代理人提交）、商标图案电子档和具体商品或服务的名称。
3. **国外自然人**：护照复印件作为身份证明文件、授权委托书（如果由代理人提交）、商标图案电子档以及具体商品或服务的名称。

请注意，具体的申请要求可能会因国家和地区政策的不同而有所变化。在实际申请前，请咨询当地的知识产权局或专业代理机构以获取最准确的信息。

看起来这个回答比单纯使用RAG的效果好很多。