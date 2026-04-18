# SP25-690-Sriramaneni
Emotion-Aware Music Recommendation Using Audio Features and Metadata
 

Abstract: In this project, I plan to build a music recommendation system that understands the emotional tone of songs. My goal is to use deep learning to identify emotions such as happy, sad, calm, or energetic from music. Many current systems recommend songs based only on user history, but they do not consider how a person feels at a particular time. I want to improve this by creating a system that predicts the emotion of a song using audio signals and metadata, and then recommends songs with similar emotions.

 

In addition to using a convolutional neural network (CNN), I will also compare it with a transformer-based model for audio understanding. This comparison will help me understand which approach works better for this task. I will evaluate how well the models predict emotions and whether the recommendations are meaningful. This project will help me understand how deep learning can be applied to real-world problems like music recommendation (Goodfellow et al., 2016).

 

Problem Statement and Motivation: I have observed that people usually listen to music based on their mood. For example, when I feel stressed, I prefer calm songs, and when I am active, I like energetic music. However, most music platforms do not fully understand emotions in songs. They mainly depend on listening history or popularity.

In this project, I aim to solve this problem by building a system that can automatically detect the emotion of a song and recommend similar songs. The task is to classify songs into emotion categories and then suggest songs based on these categories.

This problem is important because it can improve user experience in music applications. Success will be measured using accuracy and F1-score for emotion classification. For recommendations, I will check if the suggested songs belong to the same emotion category. I believe this scope is reasonable because I will use a limited dataset and a simple recommendation method.

 

Data or Environment Description: For this project, I will use publicly available datasets that contain music audio and metadata. Some datasets include emotion labels, while in some cases I may derive emotion categories from available information. These datasets are commonly used in music information retrieval research (Tzanetakis & Cook, 2002).

I will convert audio signals into features such as spectrograms or MFCC (Mel-frequency cepstral coefficients), which are widely used in audio analysis (Logan, 2000). Metadata features like tempo, genre, or energy level will also be used.

I will split the dataset into training, validation, and test sets using a 70-15-15 ratio. I will also apply basic preprocessing such as normalization and cleaning of metadata. To keep the problem simple, I will use a small number of emotion categories like happy, sad, calm, and energetic.

 

Method and Model Design: In this project, I plan to use two different deep learning approaches and compare them.

First, I will use a convolutional neural network (CNN) to extract features from audio spectrograms. CNNs are effective in capturing local patterns in audio signals (LeCun et al., 1998).

Second, I will use a transformer-based model for audio classification. The transformer can capture long-range relationships in the audio sequence and has become a strong alternative to traditional models (Vaswani et al., 2017).

For metadata, I will use a simple multilayer perceptron (MLP). The metadata features will be combined with the outputs from both CNN and transformer models to improve prediction.

As baselines, I will use:

a CNN model using only audio
a transformer model using only audio
Then I will compare them with a combined model that uses both audio and metadata.

After predicting emotions, I will build a simple recommendation system that suggests songs with similar predicted emotions. This approach is directly connected to concepts I learned in supervised learning, CNNs, and transformers.

 

Experimental Setup: I will train all models using a standard training process with validation after each epoch. I will use cross-entropy loss for classification.

To evaluate performance, I will use accuracy, precision, recall, and F1-score (Sokolova & Lapalme, 2009).

I will compare the CNN model and the transformer model to understand which performs better. I will also compare models with and without metadata.

For ablation studies, I will:

remove metadata to see its impact
compare CNN vs transformer performance
test different audio features like MFCC and spectrogram
I will tune hyperparameters like learning rate and batch size using the validation set.

 

Results: I will present results using tables and graphs comparing CNN and transformer models. I expect that the transformer model may perform better in capturing complex patterns, but CNN may still perform well due to simpler structure.

I will also use confusion matrices to understand which emotions are difficult to classify.

For the recommendation part, I will show examples where the system suggests songs with similar emotional tones. My focus will be on whether the recommendations are consistent and meaningful.

 

Failure Analysis and Limitations: I expect that both models may struggle when songs have mixed or unclear emotions. Some songs do not belong to a single emotion category, which can make classification difficult.

The transformer model may also require more data and computation compared to CNN, which can be a limitation.

The dataset may also be limited in size and diversity, which can affect performance.

I will analyze errors by checking incorrect predictions and identifying patterns. For example, the model may confuse similar emotions like calm and sad. These limitations will be discussed with proper examples.

 

Ethics, Limitations, and Responsible Use: In this project, I am not using personal or sensitive user data, but there are still some concerns. Music emotion is subjective, and different people may feel different emotions from the same song.

There is also a possibility of bias if the dataset does not include diverse types of music. This can affect fairness and generalization.

I will ensure that the system is used only as a support tool and not as a final decision-maker. It should not assume a user’s emotional state without user input. These considerations are important in responsible AI practice (Goodfellow et al., 2016).

 

Conclusion: Through this project, I aim to understand how different deep learning models, especially CNNs and transformers, can be used to detect emotions in music and improve recommendation systems. Even if the model is not perfect, it can still provide useful insights and better recommendations compared to simple methods.

This project will also help me gain practical experience with CNNs, transformers, feature extraction, multimodal learning, and evaluation techniques. It connects well with the concepts I learned in this course.

 

References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.
Logan, B. (2000). Mel frequency cepstral coefficients for music modeling. International Symposium on Music Information Retrieval.
Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437.
Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293–302.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems.