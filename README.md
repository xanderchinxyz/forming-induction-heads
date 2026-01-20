## **README:**

The goal of this project is to observe induction heads forming in real-time. To do this, I trained small, attention-only transformers on the simple task of repeating a sequence of random tokens up until the specified max context length. Because the tokens are random, the model can't cheat by learning a language; it must develop a circuit that looks at the current token, finds its previous occurrence, and predicts the token that followed it.

I trained four variations of attention-only models with absolute positional encodings to see how depth and data variety affect the formation of these circuits. Each epoch consisted of procedurally generating a batch of repeated sequences (shape: [batch_size, max_context_length]). While sequence lengths were chosen randomly between epochs for the 'varied' models, all sequences within a single batch shared a uniform length.

| Model | Layers | Dataset Type | Epochs |
| :--- | :--- | :--- | :--- |
| **1L Fixed** | 1 | Fixed length (20) | 5,000 |
| **1L Varied** | 1 | Random length (20–29) | 20,000 |
| **2L Fixed** | 2 | Fixed length (20) | 5,000 |
| **2L Varied** | 2 | Random length (20–29) | 20,000 & 60,000* |

*I extended the 2L varied run to 60000 epochs in order to see if the model further memorized or generalized to out-of-distribution (OOD sequences), hence creating two models: `2L_varied_model_20000` and `2L_varied_model_60000`.

Fixed variables:
- Vocabulary size: 256
- Model dimension (d_model): 64
- Number of heads per attention layer: 4
- Maximum context length: 80
- Batch size: 32
- Optimizer: AdamW with learning rate 0.001, betas (0.9, 0.98), weight decay 0.01

### Rough Findings:
- 1L and 2L fixed models unsurprisingly seem to have memorized the sequence length of 20, as both show fixed induction heads diagonals with intervals of 20, no matter what the sequence length is, and hence the induction accuracy (accuracy of the model after one repeated sequence) of sequences of other lengths is 0%
- The 2L fixed model seems to have developed an almost "artificial" looking blur in the heads of the first layer after the 20th token position
- Gemini mentioned that this blur could be due to the formation of a previous token head, where layer 0 acts as a helper to move information around so that layer 1 can perform the match

- 1L varied couldn't form induction heads since information or tokens can't "mix" using only one layer
- Interestingly, a blurred distribution across the 20-29 token range was formed on the attention heads in 1L varied
- Gemini believes this blurred distribution is the model's attempt to attend to the general area where the repeat is likely to be. So it's plausible that the model forms a somewhat uniform probability distribution of the tokens in the sequence, and by pure statistical chance, it can choose the correct token, leading to a low but non-zero induction accuracy

- 2L varied formed induction heads in the second layer attention heads characterised by the off diagonals at intervals of the sequence length!
- 2L varied also formed attention heads that look at the current token and previous token in the first layer
- 2L varied seems to generalize to out-of-distribution sequence lengths quite well at 20000 epochs, although not perfectly. Also, repeating tokens in a sequence sometimes confuses the 2L varied model, also trained at 20000 epochs. I believe the confusion is due to the split attention between the next token after the repeated one, hence the model can't decide which token is the correct one

- I trained the 2L varied model for another 40000 epochs for a total of 60000 epochs to see the behavior of the model on OOD sequence lengths and sequences with repeated tokens. The result is that the accuracy of OOD sequence lengths falls off afterwards, with a steeper fall off for greater sequence lengths. Perhaps the model is memorizing?
- Repeating tokens in a sequence doesn't confuse the 2L varied model trained for 60000 epochs if the sequence length is in the distribution of 20 to 30. I think this gives further proof that the model is memorizing
- Interestingly, the attention head L0H2 seems to transition from an attention head that looks at the previous token (a nice, clear long diagonal offset by one) to an attention head that looks like pure noise, similar to the attention heads when they were first initialized after training the 2L varied model further than 20000 epochs. OOD induction accuracy also starts to drop when these attention heads start fading into noise. Perhaps this head could be related to the model memorizing!

### Conclusion:
This was a very informal but fun and informative project! I would like to look more into why exactly certain patterns are formed and break down what the model is doing further, but that's out of scope for now, as this was meant to be a fun little experiment to see the formation of induction heads on 1-layer and 2-layer attention-only transformers.