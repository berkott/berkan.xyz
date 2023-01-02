---
title: 'Understanding Transformers Part 2: Technical Details of Transformers'
date: 2022-05-26
permalink: /posts/2022/05/transformersPart2/
excerpt: "Part two post in a series about understanding Transformers. This post focuses on the the technical details of Transformers."
tags:
  - Artificial Intelligence
---

*__WARNING:__ This post assumes you have a basic understanding of neural networks and specifically Recurrent Neural Networks (RNNs) and it assumes that you have read part 1.*

Now we have a solid understading of the basics of attention mechanisms, I'm going to dive right into transformers. This post will be largely based on the original Transformers paper, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf), but will also include additional intuitive explanation, technical details, and important context. 

I have not yet implemented this paper, but I found this [good PyTorch implementation](https://github.com/hyunwoongko/transformer) online. I might come back to this post and add my own implementation 🙃.

I want to start by giving relevant background to the problem this paper aims to address.

# Transduction in Machine Learning
Transfomers were originally created as transduction models in NLP. There are [several definitions](https://machinelearningmastery.com/transduction-in-machine-learning/) of transduction models, but the definition that seems most in line with how the paper uses it is "[learning to convert one string into another](Learning to Transduce with Unbounded Memory)". Neural machine translation (explained in the [last post](https://berkan.xyz/posts/2022/01/transformersPart1/)) is one of many problems that a transduction model can be used for. 

# Problems with Previous Transduction Models
Before 2017, when Transformers were introduced, the state of the art (SOTA) in transduction problems were mostly LSTMs and GRUs and their variants. The problems with these models though is that they are recursive in nature. Recurrent models require the previous hidden state in order to compute the current hidden state, meaning they are sequential in nature. This reduces computational efficiency because it prevents parallelization within training examples, which is especially important with longer input sequence lengths (training on longer articles and documents). 

There is a body of existing literature which achieves parallelization though the use of convolutional neural networks (CNNs), yet these models still struggle at learning dependencies between words that are far apart, because the number of operations required to relate words that are far apart increases with their distance. Transformers are the first transduction architecture that achieves parallelization within training examples (without convolutions or recurrence, by using attention for everything) and maintains a constant number of operations to relate words of any distance. As we will soon see, this allows transformers to achieve SOTA performance in transduction tasks while requiring less training time than previous models.

# Transformers Architecture Explained
Now for the fun part! I'm going to explain Transformers by going step-by-step through a forwards pass. I will frequently reference this very good diagram from the paper:

![Transformer Architecture](/images/transformers2/transformerArchitecture.png "Transformer Architecture")

The model resembles the encoder-decoder structure of previous transduction models (see part 1 for details). 

<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
In these green blockquotes I'm going to run through an example of translating the sentence "Bobby likes blackberries." to German, "Bobby mag Brombeeren." 
</blockquote>

# Mathematical Notation
$$\textbf{x} \in \mathbb{R}^d$$ will denote a column vector comprising of $$x_1, x_2, \ldots, x_d$$. I will often use a data matrix denoted as $$X \in \mathbb{R}^{n \times d}$$, where the $$i$$th row corresponds to the row vector $$\textbf{x}_i^\top \in \mathbb{R}^d$$. Therefore, by writing $$\textbf{x}_i$$ I am refering to a column vector of the $$i$$th row of $$X$$.

$$X \in \mathbb{R}^{n \times C}$$ and the corresponding $$\textbf{x}_i$$ vectors will represent the input words, $$H \in \mathbb{R}^{n \times d_{\text{model}}}$$ and the corresponding $$\textbf{h}_i$$ vectors will represent the hidden state, and $$\textbf{y} \in \mathbb{R}^C$$ will represent the output logits.

# Pre Encoder
<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
The input to our model is the text "Bobby likes blackberries."
</blockquote>

## Tokenization
The first challenge is to convert the input string to a meaningful representation that the model can work with. The cannonical way to do this is to assign one hot vectors to your input corpus. The only question is what unit of your input vocabulary should you assign one hot vectors to? Should each character have it's own vector? Should each word have it's own vector?

The original transformers paper, along with many other modern implementations, use [byte-pair encoding](https://arxiv.org/pdf/1508.07909.pdf) (BPE) to create subword tokens that one hot vectors can be assigned to (37000 tokens for English to German in the paper). It allows for common words to have their own representation, but breaks rare words into multiple subtokens in order to include all words in the corpus without requiring an incredibly large vocabulary size. Here is how BPE works (visit [this page](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0) for a more detailed walkthrough of this algorithm.):

1. Split up the entire corpus into characters and add a special start of sentence token &lt;SOS&gt;, end of word token &lt;EOW&gt;, and end of sentence token &lt;EOS&gt;. This is the initial token list. 
2. Count all token pairs and replace the individual tokens of the most common pair with the pair itself.
3. Repeat step 2 until the desired vocabulary size, $$C$$, is reached.

Now that we have split up the input sentence into meaningful tokens, we can create one hot vectors from these tokens. I will be using the following notation to represent one hot vectors: $\textbf{e}_i \in \mathbb{R}^C$, where $\textbf{e}_i$ is the $i$th standard basis column vector. In other words, $\textbf{e}_i$ represents a one hot vector where the value corresponding to the $i$th dimension is 1 and the rest are 0. 

<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
After tokenization, our input sentence "Bobby likes blackberries." is turned into the following list of tokens ["&lt;SOS&gt;", "Bobby&lt;EOW&gt;", "likes&lt;EOW&gt;", "black", "berries&lt;EOW&gt;", "&lt;EOS&gt;"] using BPE with vocabulary size $C = 37000$. Now we can vectorize this list to get $\begin{bmatrix} \textbf{e}_{41}^\top \\ \vdots \end{bmatrix} = \begin{bmatrix} \textbf{x}_1^\top \\ \vdots \end{bmatrix} = X \in \mathbb{R}^{n \times C}$, where $\textbf{x}_i$ is the one hot vector for the $i$th input word and $n$ is called the context size. More details on the value of n can be found in the decoder section.

<br><br>

(Please note that the output tokens from BPE and your one hot vectors depend on your input and I'm just arbitrarily choosing plausible values.)
</blockquote>

## Word Embeddings

But these one hot input vectors are still very high dimensional. In fact, we can embbed these vectors into a lower dimensional vector space that is easier for the model to work with. To do this, the original transformers paper uses weight sharing of the input embeddings, output embeddings, and unembedding to get the logits, similar to what was done by [Press et al.](https://arxiv.org/pdf/1608.05859.pdf) (see the paper for more details, I will just give a brief explanation). 

A word embedding matrix $$U \in \mathbb{R}^{d_{\text{model}} \times C}$$ ($$d_{\text{model}}$$ is just the dimension of many word representations in the model, it is $$512$$ in the original transformers paper) is used to convert $$\textbf{x}_i$$ into a lower dimensional vector $$ U \textbf{x}_i \in \mathbb{R}^{d_{\text{model}}}$$ that matches the model size and represents the properties of the input word. For more information about word embeddings please see these resources (TODO: UPDATE THESE RESOURCES). We will use the same matrix $$U$$ to unembed our hidden state to get the logits (see unembedding section). We train $$U$$ along with the rest of our network (no pretrained word embeddings used). 

<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
Now we can take our list $\begin{bmatrix} \textbf{x}_1^\top \\ \vdots \end{bmatrix} = X$ of token one hot vectors, and compute the word embeddings $\begin{bmatrix} (U \textbf{x}_1)^\top \\ \vdots \end{bmatrix} = X U^\top \in \mathbb{R}^{n \times d_{\text{model}}}$.
</blockquote>

## Positional Encodings
Because transformers don't have any sort of recurrence or convolutions, the model has no way of understanding the order of the inputs. The model needs some form of positional encoding of the input tokens. In the original transformers paper, these encodings were explicitly given and not learned (empirical evidence suggests learned embeddings didn't perform better in general and even performed worse for input sequences longer than the ones in the training data).

$$ PE_{(pos, 2i)} = \sin(pos/1000^{2i/d_{\text{model}}}) $$

$$ PE_{(pos, 2i + 1)} = \cos(pos/1000^{2i/d_{\text{model}}}) $$

$$ PE = \begin{bmatrix}
          \cos(1/1000^{1/d_{\text{model}}}) & \sin(1/1000^{2/d_{\text{model}}}) & \cdots\\
          \cos(2/1000^{1/d_{\text{model}}}) & \sin(2/1000^{2/d_{\text{model}}}) & \\
          \vdots &  & \ddots
        \end{bmatrix} \in \mathbb{R}^{n \times d_{\text{model}}} $$

Where $$pos$$ is the position of the token and $$i$$ is a dimension in that token's embedding. This is what the encodings partially look like (image taken from the illustrated transformer CITE!!!).

![Positional Encodings](/images/transformers2/attention-is-all-you-need-positional-encoding.png "Positional Encodings")

The authors claim they used these positional encodings because they hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $$k$$, $$PE_{pos+k}$$ can be represented as a linear function of $$PE_{pos}$$. These positional encodings get summed with the original embeddings to create the input into the transformer encoder. I'm going to use $$H \in \mathbb{R}^{n \times d_{\text{model}}}$$ to indicate hidden state of the model (note that the dimension of the hidden state never changes in any of the transformer layers).

$$ H = X U^\top + PE $$

<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
We can represent our input sentence as $H = X U^\top + PE \in \mathbb{R}^{n \times d_{\text{model}}}$, where we have one column vector that represents for each input token and its position.
</blockquote>

![Pre Encoder](/images/transformers2/preEncoder.png "Pre Encoder")

# Encoder

Now we are finally ready to dive into the encoder part of the model. In the original paper, the encoder consists of a stack of 6 transformer layers, each with two sublayers. The first sublayer is multi-head attention, and the second is just a multilayer perceptron (MLP). Additionally, there is a residual connection between each of these layers, along with layer normalization.

## Residual Stream
A nice way to think about these models, inspired by [Elhage et al.](https://transformer-circuits.pub/2021/framework/index.html#residual-comms), is that we have a stream of data that is being added to by the multi-head attention and MLPs and is normalized by the layer normalizations. This stream is almost entirely linear, and it serves as an important communications channel throughout the model. The dimensions of this stream are also never changed throughout the entire model.

![Residual Stream in Transformer](/images/transformers2/anthropicTransformer.png "Residual Stream in Transformer")

## Multi-Head Attention
We can think of the function of multi-head attention as [moving information](https://transformer-circuits.pub/2021/framework/index.html#residual-comms) in the residual stream and applying some weight matrices. Let's start with a single attention head:

### Scaled Dot Product Attention for One Attention Head
The transformers architecture implements scaled dot product attention. This is in contrast to additive attention, which was described in the [last post](https://berkan.xyz/posts/2022/01/transformersPart1/). In additive attention, we have an entire alignment model (a small MLP) that computes the logits that get fed into the softmax to determine the weighting of the values. In dot product attention, we replace that alignment model with dot products!

If you read the original transformers paper, you will see scaled dot product attention written in the most general form as:

$$ \text{Attention}(Q, K, V) = \text{softmax} \big( \frac{QK^\top}{\sqrt{d_k}} \big) V $$

<!-- Lets break down what's actually going on. In practice, we actually have $$h=8$$ different attention heads, each with their own query, key, and value matrices. These are calculated from the hidden state as follows:  In practice , so  prefer to write it as follows:

$$ \text{head}_i = \text{softmax} \big( \frac{Q_iK^\top_i}{\sqrt{d_k}} \big) V_i $$

The $$Q_i, K_i, V_i$$, represent a query, key, and value matrix respectively that are unique for each attention head. They are simply derived from the hidden state as follows.  -->

Let's ignore what the Q, K, and V matricies represent for the moment. Initially, this notation was a bit confusing for me at first because, technically the softmax function is a vector function. The key insight is that the softmax acts on the rows of the $$Q K^\top$$ matrix. Things are more clear when they are written as follows:

$$
\text{Attention}(Q, K, V) = \begin{bmatrix}
  \text{softmax}\big(\frac{\textbf{q}_i^\top K^\top}{\sqrt{d_k}}\big) V \\
  \vdots
\end{bmatrix} 
$$

Essentially, we are just taking some convex combination of the rows of $$V$$ in each row of our attention matrix. 

### Relation to Additive Attention
As a refresher, additive attention from the previous post looked like this:

$$ \textbf{c}_i = \sum_{j=1}^{T_x} \alpha_{ij} \textbf{h}_j $$

$$ \alpha_{ij} = \text{softmax}(\textbf{e}_{i})_j $$

$$ \textbf{e}_{ij} = a(\textbf{s}_{i-1}, \textbf{h}_j) $$

Where $$\textbf{c}_i$$ was the $$i$$th context vector, $$a$$ was the alignment model, $$\textbf{s}_{i-1}$$ was the previous hidden state of the decoder, and $$\textbf{h}_j$$ was the $$j$$th output from the bidirectional RNN encoder. We can summarize the equations above as follows (note that the $$T_x = n$$ from the previous post and just represents the input length): 

$$ \textbf{c}_i = \sum_{j=1}^{n} \text{softmax}(a(\textbf{s}_{i-1}, H))_j \textbf{h}_j = \text{softmax}(a(\textbf{s}_{i-1}, H)) H $$

If we take $$Q = s_{i-1}$$ and $$K = V = H$$ we can write:

$$ \textbf{c}_i = \text{softmax}(a(\textbf{s}_{i-1}, H)) H = \text{softmax}(a(\textbf{q}_i, K)) V $$

In principle, all both of these attention mechanisms do is take convex combinations of the rows of some value matrix! The main difference is just how the convex combination weights are calcuated. Additive attention uses a small MLP alignment model, and dot product attention uses dot products. In fact, if $$a(\textbf{x}, Y) = \frac{\textbf{x}^\top Y^\top}{\sqrt{d_k}}$$ the formula above exactly describes scaled dot product attention for one token.  The transformer paper claims that while these are similar in theoretical complexity, dot-product attention is much faster and more space efficient on modern hardware. I find it incredible how we can basically just replace a small neural network with a dot product and maintain good performance 🤯. 

### Why Scaled Dot-Product Attention?
You might be wondering why this is called __scaled__ dot-product attention. That is because of the $$\frac{1}{\sqrt{d_k}}$$ term in the softmax. The paper claims this is because additive attention outperforms dot product attention for large values of $$d_k$$ without the scaling. They suspect this is becuase the dot products grow large in magnitude, pushing the softmax function into regions of [extremely small gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), which slows gradient descent. Let's disect why this is true, why do the dot products grow large in magnitude and why does this push the softmax function into regions of small gradients?

We can think of a dot product between two vectors of dimension $$d_k$$ as $$v \cdot w = \sum_i v_i w_i$$. If each entry in these vectors is a random variable with mean 0 and variance 1, then because variances add, the dot product will have mean 0 and variance $$d_k$$. This is of course an oversimplification because the entires of our queries and keys will not be random variables with mean 0 and variance 1, but it illustrates the fact that as $$d_k$$ grows, so does the variance of our dot product. Now how does this impact the gradient of the softmax function (technically gradients are only defined for scalar functions, so from now on I will refer to the [derivative of the softmax function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)). Here is the derivative of the $$i$$th output with respect to the $$j$$th input.

$$ \text{softmax}(v)_i = \frac{e^{v_i}}{\sum_{k=1}^N e^{v_k}}, \ v \in \mathbb{R}^N $$

$$
  \frac{\partial \text{softmax}(v)_i}{\partial v_j} = \begin{cases} 
    \text{softmax}(v)_i (1 - \text{softmax}(v)_j) & \text{if } i = j \\
    - \text{softmax}(v)_i \text{softmax}(v)_j & \text{if }  i \neq j 
  \end{cases}
$$

Due to the exponent, when there is a large term in the input vector to the softmax, it ends up dominating the output probability distribution. Therefore, if we have large dot products from before, the input to the softmax is going to have some terms that are very large. In both cases for the derivative, it is likely that either $$\text{softmax}(v)_i$$ or $$\text{softmax}(v)_j$$ will be small, making the overall derivative also very small.

However, by adding the scaling term that is proportional to $$\sqrt{d_k}$$, where $$d_k$$ is the dimension of the dotted vectors, we make the values in the input vector to the softmax less extreme, and therefore make it less likely that the overall derivative is very small.

## Back to Multi-Head Attention
Instead of just using one attention head as in the original attention paper, the authors found it beneficial to linearly project the queries, keys, and values $$h$$ times to a lower dimensional space, run attention on each, and aggregate the results.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots \text{head}_h) W^O $$

$$ \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$

Where $$ W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k} $$, $$ W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k} $$, $$ W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v} $$, and $$ W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}} $$. Additionally, they set $$h=8$$ and $$d_k = d_v = d_{\text{model}}/h = 64$$. The computational cost of this is similar to if they had just used a single head attention with full dimensionality.

## Multi-Head Self-Attention in the Encoder
The way this multi-head attention plays out in the encoder is as self-attention, meaning that the [queries, keys, and values](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) all are equal to the hidden state $$H$$. In total, this looks as follows:

![Multi-Head Self-Attention](/images/transformers2/multiHeadSelfAttention.png "Multi-Head Self-Attention")

One reasonable question is, why do we even need multi-head self-attention? One thing attention gets you is the ability to mix information between tokens. This is the only part of the architecture that has this capability

## Layer Normalization
[Layer normalization](https://arxiv.org/pdf/1607.06450.pdf) is a commonly used method to reduce training time by normalizing hidden states within a single training example. Nowadays, it seems [more commonly used](https://paperswithcode.com/method/layer-normalization) than [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) because it is able to be used for online learning tasks or places where it is impossible to have large mini-batches, and is difficult to apply to RNNs. Here is a [mathematical explanation](https://leimao.github.io/blog/Layer-Normalization/) of how it works in transformers:

First we compute the mean and variance of each of the n tokens of the our training (or test) sample:

$$ \mu_{t} = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} \textbf{h}_{t, i} $$

$$ \sigma_{t} = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (\textbf{h}_{t, i} - \mu_{t})^2} $$

Where $$\mu_{t}$$ represents the mean for token $$t$$ and $$H_{t, i}$$

The rest of layer normlization was quite suprising to me, we also add a $$\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$$ term that gets applied as follows:

$$\text{LayerNorm}(\textbf{h}_{t}) = \gamma \frac{\textbf{h}_{t} - \mu_{t}}{\sigma_{t} + \epsilon} + \beta$$

![LayerNorm](/images/transformers2/layerNorm.png "LayerNorm")

## MLP
The MLP layers of the transformer are applied identically to each token in the hidden state, and are added back into the residual stream. These layers are essentially comprised of two affine transformations with a ReLU ([GeLUs](https://arxiv.org/pdf/1606.08415.pdf) are usually used in modern models, but not in the original paper) in the middle. This can be written as follows:

$$\text{MLP}(\textbf{h}_i) = \text{ReLU}(\textbf{h}_i W_1 + \textbf{b}_1) W_2 + \textbf{b}_2$$

One obvious question might be, what do these MLPs do? [Recent work from Anthropic](https://transformer-circuits.pub/2022/solu/index.html) suggests that in language models neurons in MLPs may represent certain rules or categories such as phrases related to music or base64-encoded text. From my own work, it seems that, at least in early layers, of Vision Transformers, MLP neurons often correspond to certain visual textures or patterns.

![MLP](/images/transformers2/mlp.png "MLP")

## Layer Normalization
After the MLP, there is another LayerNorm, which is applied identically to the one before.

## Final Encoder Output
What was described above was one encoder block. Note that the input dimensions and output dimensions of each transformer block are the same, namely $$H \in \mathbb{R}^{n \times d_{\text{model}}}$$. Therefore, we can just take the output of our first block, and feed it into our second block. This is done $$N$$ times. The final hidden state that comes out of the encoder blocks then becomes the key and value matrices for the decoder $$K_E, V_E = H$$, more on this later. The encoder is only ran once.

Here is the full encoder:
![Encoder](/images/transformers2/encoder.png "Encoder")

# Pre Decoder
In the original transformers paper, the decoder is used autoregressively, meaning that output translations are fed back into the model. Initially, nothing except for a start of sentence token &lt;SOS&gt; is fed into the model. But as the model starts making predictions, the input sequence to the decoder grows. The input into the encoder is processed the same way as the input to the encoder (see the pre-encoder section).

<blockquote style="background-color: #ecfeec; border-color: #528852; font-style: inherit;">
Our current output sentence is "Bobby mag", which is tokenized as ["&lt;SOS&gt;", "Bobby&lt;EOW&gt;", "mag&lt;EOW&gt;"] using BPE with vocabulary size $C = 37000$. Now we can vectorize this list to get $\begin{bmatrix} \textbf{e}_{41}^\top \\ \vdots \end{bmatrix} = \begin{bmatrix} \textbf{x}_1^\top \\ \vdots \end{bmatrix} = X \in \mathbb{R}^{n \times C}$, where $\textbf{x}_i$ is the one hot vector for the $i$th input word.

<br><br>

(Please note that the output tokens from BPE and your one hot vectors depend on your input and I'm just arbitrarily choosing plausible values.)
</blockquote>

![Pre Decoder](/images/transformers2/preDecoder.png "Pre Decoder")

# Decoder
The decoder is almost exactly the same as the encoder, except the attention works differently. Instead of just one multi-head self-attention block, we have a masked multi-head self-attention block, another layernorm, and a multi-head attention block.

## Masked Multi-Head Self-Attention
Instead of using regular mult-head self-attention, the decoder uses masked multi-head self-attention. Masked multi-head self-attention doesn't allow for tokens to attend to later tokens. This is implemented by making out (setting to $$-\infty$$) all values in the input of the softmax which would cause attention to later tokens. This just requires us to slightly modify our equation for attention and add in a "look ahead masks".

$$ \text{MaskedAttention}(Q, K, V) = \text{softmax} \Big( \frac{QK^\top}{\sqrt{d_k}} + 

\begin{bmatrix}
  0 & -\infty & \cdots & -\infty \\
  \vdots & 0 & -\infty & \vdots \\
  0 & 0 & 0 & -\infty \\
  0 & 0 & \cdots & 0
\end{bmatrix} 

\Big) V $$

The paper claims masking is necessary "to prevent leftward information flow in the decoder to preserve the auto-regressive property." To be more precise, I think the word "prevent" should be replaced with "reduce". The reason being that technically, even for the first attention head, the tokens of the value matrix don't have to precisely correspond to tokens. This is because the value weight matrix applies some linear transformation to the hidden state. This transformation could mix information between tokens, such that even if it is impossible for the softmax to produce connections to later tokens, the tokens representations themselves could have information from later tokens. That being said, in practice, it seems that this still enables the transformer to preserve auto-regressive property because the model works!

If we didn't train in parallel, we actually wouldn't need masking at all. [This is because](https://stackoverflow.com/questions/58127059/how-to-understand-masked-multi-head-attention-in-transformer) we could technically do all our training recurrently, where with each token we autoregressively generate, we save the hidden state for that token, and then use it when generating the next token (more on this later).

## Multi-Head Attention
This multi-head attention block is almost the same as the multi-head self-attention in the encoder, except that the key and value matrices come from the output of the encoder. Taking the keys and values from the encoder is necessary in order to incorporate information from the the input sentence into the output translation.

Note, because the value matrix is not the current decoder hidden state $$H$$, if the number of tokens in the encoder and decoder were different, we would not be able to add back into the residual stream! Therefore, in practice, a default max number of tokens, or context size, $$n$$ is set (to 128 for instance) for both the encoder and decoder blocks. If the sequence is not $$n$$ tokens long, the remaining tokens are filled in by padding &lt;PAD&gt; tokens. 

# Post Decoder
Just like with the encoder, there are $$N$$ of decoder blocks. To get our output logits using the hidden state $$H$$ passed through the transformer decoder layers, we use the embedding matrix $$U$$ to unembedd the hidden representation. Our logits are the $$ U^\top h_n = y \in \mathbb{R}^C$$, which we can pass through a softmax to get the output probability distribution over our vocabulary. Note how we choose the final, or $$n$$th token vector to get our output logits. This is because we are predicting the next word.

The decoder is run autoregressively until it outputs the &lt;EOS&gt; token, at which point the translation is finished!

# Training a Transformer


# Variants on the Original Architecture

## Decoder Only Autoregressive Transformers

GPT Models: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
Decoder only autoregressive transformer I think: https://arxiv.org/pdf/1801.10198.pdf

http://nlp.seas.harvard.edu/annotated-transformer/

https://johnthickstun.com/docs/transformers.pdf

# Conclusion

Discussion over performance: Why are transformers better? How are inputs embedded into transformers (maybe include this in the next post)?

[add blurb about transformers]. The next post will focus on the uses of Transformers in modern deep learning and speculation of how far scaling Transformers can go. Stay tuned 😎. 

# References:
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [BLEU Score](https://aclanthology.org/P02-1040.pdf)
- [Someone Else's Implementation](https://github.com/hyunwoongko/transformer)

http://jalammar.github.io/illustrated-transformer/
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
