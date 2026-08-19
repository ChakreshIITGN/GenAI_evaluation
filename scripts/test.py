#%%
from ollama import chat

response = chat(
    model='gpt-oss:120b-cloud',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)
# %%
