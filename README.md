# GPT_from_scratch
The souces of this repo is from Andrej Karpathhy's [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [repo](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

Before we jump into implememting GPT, we can into [bigram.py](notebooks/bigram.py) to understand how a simple bigram model works that given a token gererate another token. To build a GPT we used Shakespear's text [input.txt](notebooks/input.txt) that contains roughly 1M characters. Since we are using character based token to build a GPT hence its 1M tokens. However, ChatGPT uses sub-word token (tiktoken). In subword token it will turn out rooughly 300K tokens.

The main building block of GPT is a transformer head that is self attention block and it uses scaled dot product. At first it excutes a dot product of `query` and `key` to get the weight and apply masking and softmax. The masking is applied to prevent a token to see the next tokens. Then the weight is nomalized by dividing with `√emb` to make it unit gaussian

After implementing a tranformer head we can implement multihead then multihead + feed-forward network and then attention-block with skip connection or residual network. Here is the break down of loss-function (cross-entropy):

**Model -- Parameter -- Loss function value**
* single self-attention head -- 0.5MM --2.26
* Multi-head -- 0.73MM -- 2.03
* Multi-head + ffwd nn -- 1.92MM --1.67
* Attention block + Res+ Layer Norm -- 10MM - 1.56

So far we have discussed about pre-traing stage . After that there is fine-tunning stage . For chat-gpt, there are three steps for fine-tunning:
* Collect demonstration data and train a supervised policy
* Collect comparison data by human raters and train a reward model
* Optimize policy against the reward model using the PPO/DPO reinforcement learning algo.

For more detains see the OpenAI [blog](https://openai.com/index/chatgpt/)
