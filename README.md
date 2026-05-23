# Do Transformers Learn to Sample? Comparing Autoregressive and Masked Models on Probability Distributions

## Central research question

When trained on samples from known probability distributions, do autoregressive and masked Transformers differ in:
1. how accurately they generate the target distribution;
1. how independent their generated samples are;
1. how they internally represent distributional properties such as mean, variance, modality and entropy?

The motivation comes directly from Large Language Models Are Bad Dice Players, which evaluated prompted frontier LLM outputs and found poor distributional sampling, especially across stateless independent requests. Your study would move from behavioural evaluation to controlled training and mechanistic analysis


## Project Structure:: 

1. Data preparation: build distributions and evaluation metrics 
2. Casual transformer: build a small transformer and train it on probability distributions. 
3. Test different kinds of masking and sampling (think more about diffusion)
4. Compare hidden representations with the one of frontier LLMs on the same task. 



