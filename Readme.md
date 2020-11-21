# COVID-19 Detection with 92.14% Validation Set Accuracy

## Goal
The goal is to use the dataset available [here](https://github.com/ieee8023/covid-chestxray-dataset) to develop AI based approaches to predict and understand the COVID-19 infection.  

## Expected Outcomes
This would give physicians an edge and allow them to act with more confidence while they wait for the analysis of a radiologist by having a digital second opinion confirm their assessment of a patient's condition. Also, these tools can provide quantitative scores to consider and use in studies.

## About the dataset
The dataset is a public open dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS). Data is collected from various public sources as well as through indirect collection from hospitals and physicians. All images and data is released publicly in [this](https://github.com/ieee8023/covid-chestxray-dataset) GitHub repo.

The dataset contains a `metadata.csv` file. Here is a list of each metadata field, with explanations where relevant

| Attribute | Description |
|------|-----|
| patientid | Internal identifier |
| offset | Number of days since the start of symptoms or hospitalization for each image. If a report indicates "after a few days", then 5 days is assumed. This is very important to have when there are multiple images for the same patient to track progression. |
| sex | Male (M), Female (F), or blank |
| age | Age of the patient in years |
| finding | Type of pneumonia |
| RT_PCR_positive | Yes (Y) or no (N) or blank if not reported/taken |
| survival | Yes (Y) or no (N) or blank if unknown|
| intubated | Yes (Y) if the patient was intubated (or ventilated) at any point during this illness or No (N) or blank if unknown. |
| went_icu | Yes (Y) if the patient was in the ICU (intensive care unit) or CCU (critical care unit) at any point during this illness or No (N) or blank if unknown.|
| needed_supplemental_O2 | Yes (Y) if the patient required supplemental oxygen at any point during this illness or No (N) or blank if unknown |
| extubated | Yes (Y) if the patient was successfully extubated or No (N) or blank if unknown |
| temperature | Temperature of the patient in Celsius at the time of the image|
| pO2 saturation | partial pressure of oxygen saturation in % at the time of the image |
| wbc count | white blood cell count in units of 10^3/uL at the time of the image |
| neutrophil count | neutrophil cell count in units of 10^3/uL at the time of the image |
| lymphocyte count | lymphocyte cell count in units of 10^3/uL at the time of the image |
| view | Posteroanterior (PA), Anteroposterior (AP), AP Supine (APS), or Lateral (L) for X-rays; Axial or Coronal for CT scans. Translations: Bettaufnahme->Supine, Liegend->Supine|
| modality | CT, X-ray, or something else |
| date | Date on which the image was acquired |
| location | Hospital name, city, state, country |
| filename | Name with extension |
| doi | Digital object identifier ([DOI](https://en.wikipedia.org/wiki/Digital_object_identifier)) of the research article |
| url | URL of the paper or website where the image came from |
| license | License of the image such as CC BY-NC-SA. Blank if unknown |
| clinical notes | Clinical notes about the image and/or the patient |
| other notes | e.g. credit |  

## Approach
In the dataset, one can categorise the features into three sets:
1. Visual feature depicting image
2. Textual feature depicting clinical notes describing the condition of patient
3. Additional features describing the patient  

#### Main idea -
The main idea is to train three classifiers for each of the features, and then implement the concept of  ensemble learning to combine these models' predictions and output the final prediction. Currently, I’ve given equal weights to each classifier but this can be changed in further experiments.

#### For visual features -
- For visual features, I’ve first loaded the images using OpenCV and resized them to 600 x 600 x 3 dimensions. Then I’ve passed these images through EfficientNetB7 architecture to generate image embeddings.  
- Generally, the models are made too wide, deep, or with a very high resolution. Increasing these characteristics helps the model initially but it quickly saturates and the model made just has more parameters and is therefore not efficient. In EfficientNet they are scaled in a more principled way i.e. gradually everything is increased.
- Each image embedding has a dimension of 2560. Then I’ve passed these image embeddings through a Random Forest Classifier to predict if the patient has COVID-19 infection or not.

#### For textual features -
- For textual features i.e., clinical notes describing the condition of a patient, I thought this is an important feature. Because as I went through the clinical notes, I could sense the information they contain definitely have some meaning and is useful to predict if a patient has COVID-19 or not. Now the only task was to convert these clinical notes into some kind of vector representation.
- There are many models for this, from word2vec, glove to BERT and RoBERTa. But I chose the BioSentVec model.
- [BioSentVec](https://github.com/ncbi-nlp/BioSentVec#-biosentvec-2-biomedical-sentence-embeddings-with-sent2vec) is an extension of original BioWordVec trained using [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/) and [MIMIC - III](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiR1-XYid_sAhUkyDgGHXZEB3cQFjAAegQIARAC&url=https%3A%2F%2Fmimic.physionet.org%2F&usg=AOvVaw3DHMAC4-_K1Sd8Gj2Dv0In).
- PubMed delivers a publicly available search interface for MEDLINE as well as other NLM (U.S. National Library of Medicine) resources, making it the premier source for biomedical literature and one of the most widely accessible resources in the world.
- MEDLINE (Medical Literature Analysis and Retrieval System Online, or MEDLARS Online) is a bibliographic database of life sciences and biomedical information. It includes bibliographic information for articles from academic journals covering medicine, nursing, pharmacy, dentistry, veterinary medicine, and health care.
- MIMIC-III (‘Medical Information Mart for Intensive Care’) is a large, single-center database comprising information relating to patients admitted to critical care units at a large tertiary care hospital. Data includes vital signs, medications, laboratory measurements, observations and notes charted by care providers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital length of stay, survival data, and more.
- [BioSentVec model](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin) computes 700-dimenstional sentence embeddings. BioSentVec model is trained using [sent2vec](https://github.com/epfml/sent2vec) library that provides numerical representations (features) for words, short texts, or sentences, which can be used as input to any machine learning task, with bigram model, window size equal to 20 and negative examples 10.
- There were some samples for which there were no clinical notes, so I filled the missing values with a sentence “There is no clinical note for this patient”. I’d the choice to fill them up with all zeros but that wouldn’t go with other samples. So to make the distribution similar I used this sentence. In further experiments we can replace the missing values from a set of random sentences.
- Each clinical note embedding has a dimension of 700. Then I’ve passed these image embeddings through a Random Forest Classifier to predict if the patient has COVID-19 or not.

#### For additional features -
- For additional features in the dataset like age, offset, sex, etc. I consulted a doctor and asked his help to make sure what features are important for detecting COVID-19 and what features are just adding noise to the data. As seen in the architecture, I’ve summarised a list of 9 features namely - sex, age, RT PCR Positive, intubated, intubation present, went ICU, in ICU, needed supplemental O2, and view.
- For offset and age features, I've replaced the missing values with the most occurring value as it shows the general trend of most patients admitted. And to rest all other features, I've replaced missing values with a tag 'Unclear'. Also I've transformed age feature into a categorical feature. This is because ages like 23, 24 or 25 won't make much difference. So I've divided them into appropriate bins.
- Then I've one-hot encoded them and trained a Random Forest Classifier to predict if the patient has COVID-19 or not.  

#### For final module -
Now after having each classifier’s prediction, I’ve combined them to predict the final prediction. The methodology used to combine them is - if two or more classifiers have output 1 then the final prediction is 1, else the final prediction is 0.  

## Future Scope of Work
- As said earlier also, we can give weights to each classifier's predictions and then generate the final prediction.
- We can replace missing clinical notes with some random sentences.
- Also, we can try normalizing the input images as an experiment and see if it further improves the accuracy or not.
- Instead of EfficientNetB7 we can give a try to other state-of-the-art architectures also.
- Also instead of using random forest classifiers for each category we can experiment with different classifiers and using different classifiers for each category.
