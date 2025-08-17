import os                           # operating-system utilities
import io                           # in-memory streams (for uploaded files)
import sys                          # system-specific parameters and functions
import math                         # math helpers
import time                         # time utilities for simple timing
from pathlib import Path            # easier path handling
import pandas as pd                 # main dataframe library
import numpy as np                  # numerical arrays
import matplotlib.pyplot as plt     # plotting (used for ROC, wordcloud display)
from wordcloud import WordCloud      # word cloud generation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,roc_curve, confusion_matrix, classification_report)
import joblib                       # saving/loading models mostly for external use
import streamlit as st              # the Streamlit library
import kagglehub
from kagglehub import KaggleDatasetAdapter

#Streamlit page configuration
st.set_page_config(page_title='Emotion WordClouds (TF-IDF) — Amazon Reviews', layout='wide')

# Small helper: get file size in megabytes (approx)
def sizeof_mb(obj_bytes):
    """Return size in megabytes for a bytes-like object or path."""
    if isinstance(obj_bytes, (bytes, bytearray)):
        return len(obj_bytes) / (1024 * 1024)
    try:
        return os.path.getsize(obj_bytes) / (1024 * 1024)
    except Exception:
        return None

# Caching IO-heavy steps: use st.cache_data to avoid re-running expensive file reads / preprocessing during the same user session.
@st.cache_data(show_spinner=False)
def read_csv_safe(file_buffer, nrows=None):
    """Read CSV robustly from an uploaded file-like object or path.
    - file_buffer: either a path (str / Path) or a file-like object (BytesIO)
    - nrows: optionally read only the first nrows (useful for trimming large CSVs)
    """
    # pandas will accept a path-like or file-like object for read_csv
    if nrows is not None:
        return pd.read_csv(file_buffer, nrows=nrows)
    return pd.read_csv(file_buffer)

# Define the list of pages and their labels.
PAGES = [
    'Overview',
    'Load & Preprocess Data',
    'Features & Word Clouds',
    'Modeling & Evaluation',
    'Manual Sentiment Test',
    'Deployment & Notes'
]

st.sidebar.title('Pages') # Sidebar contains the page navigation.
page = st.sidebar.radio('', PAGES)

if page == 'Overview':
    st.title('Emotion-Specific Word Clouds — Amazon Fine Food Reviews (TF-IDF)')
    # Short description / instructions (visible only on Overview)
    st.markdown(
        """
        **What this app does**

        - Creates TF-IDF embeddings from raw review text.
        - Generates emotion-specific word clouds (we map review star ratings to
          simple emotion labels for visualization).
        - Trains two classifiers (Logistic Regression and Random Forest) to
          predict the coarse emotion label from text features.
        - Evaluates using Precision, Recall, F1-score, and ROC-AUC.

        **Navigation:** Use the sidebar to move between pages, everything on a page has to be satisfied on a page before continuing to the next. 
        """
    )
    st.info('Use the sidebar to go to Upload & Preprocess to start.')

#Upload & Preprocess Page (Page 2)
if page == 'Load & Preprocess Data':
    st.header('Load & Preprocess Data')
    st.write('Once you download the "Reviews.csv" dataset from Kaggle and place it in the parent folder, load the data')

    local_default_path = 'Reviews.csv'
    use_local = False
    if os.path.exists(local_default_path):
        use_local = st.checkbox('Use local app file') # developer-provided upload path

    file_buffer = None # Use local file, set buffer to path string; else use uploaded file buffer
    source_desc = '' #makes it readable
    if use_local and 'df_raw' not in st.session_state:
        file_buffer = local_default_path
        source_desc = f'local file at {local_default_path}' #creates a string showing the file path

        approx_mb = None # Attempt to determine approximate file size. If uploaded file, it's in-memory
        if isinstance(file_buffer, (str, Path)):
            approx_mb = sizeof_mb(str(file_buffer))
        else:
            try:
                pos = file_buffer.tell() # Uploaded file is a BytesIO-like object
            except Exception:
                approx_mb = None

        st.write(f'Using dataset from **{source_desc}**') # shows file being used
        if approx_mb:
            st.write(f'Approx. file size: **{approx_mb:.1f} MB**')

        MAX_MB = 200  # Streamlit's upload limit is often 200MB. If the file is larger, pick samples from large dataset
        df = None
        # If the file seems larger than MAX_MB and it's an upload, sample rows
        if approx_mb and approx_mb > MAX_MB and not isinstance(file_buffer, str):
            df = read_csv_safe(file_buffer)
            st.success(f'Read {len(df):,} rows (sampled from the large upload).') #confirms successful read with actual row count
        else:
            try:
                df = read_csv_safe(file_buffer) # Normal reading either local path or under 200mb sized upload
            except Exception as e:
                st.error('Failed to read CSV automatically. Attempting a more robust read...')
                try:
                    if isinstance(file_buffer, (str, Path)):
                        df = pd.read_csv(file_buffer, encoding='latin-1') # try with latin-1 encoding fallback
                    else:
                        file_buffer.seek(0)
                        df = pd.read_csv(file_buffer, encoding='latin-1')
                except Exception as e2:
                    st.error(f'Failed to read CSV: {e2}')
                    df = None #sets df to none if the reading fails

        if df is not None:
            st.session_state['df_raw'] = df  # ✅ Store raw data in session state
            st.success('Data loaded and stored in memory. Ready for preprocessing.') # If we obtained a dataframe, show a small preview and allow selecting columns

    # Show preview if data is loaded
    if 'df_raw' in st.session_state:
        st.write('Preview of the dataset (first 5 rows):')
        st.dataframe(st.session_state['df_raw'].head())

        expected_cols = ['Text', 'Score']
        missing = [c for c in expected_cols if c not in st.session_state['df_raw'].columns]
        if missing:
            st.error(f'The uploaded dataset is missing columns: {missing}. Please ensure it matches the Amazon Fine Food Reviews schema.')
        else:
            st.success('Required columns found: Text and Score.')

            st.subheader('Preprocessing options') # Simple preprocessing choices visible to the user
            lowercase = st.checkbox('lowercase text (recommended for model training)', value=True)
            remove_na = st.checkbox('drop rows with missing Text or Score', value=True)
            sample_frac = st.slider('Randomly sample fraction of rows for quick experiments', 0.01, 1.0, 1.0)

            if st.button('Apply preprocessing'): # Apply preprocessing when user clicks the button
                df = st.session_state['df_raw'].copy() # creates copy to preserve the original data since we just use random samples
                if remove_na:
                    df = df.dropna(subset=['Text', 'Score']) # removes rows where there is nothing in 'Text' and 'Score' column
                if lowercase:
                    df['Text'] = df['Text'].astype(str).str.lower()
                if sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True) #if less than 100% selected, randomly select rows
                else:
                    df = df.reset_index(drop=True) # after selecting, the index is reset to be able to reselect for another user

                st.session_state['df_working'] = df # store in streamlit's session state for other pages
                st.success(f'Preprocessing applied. Dataset now has {len(df):,} rows.') # store in streamlit's session state for other pages
                st.write('Stored preprocessed dataframe in session state as `df_working`. Move to the Features & Word Clouds page.') #guides user to the next step

# Features & Word Clouds Page (Page 3)
if page == 'Features & Word Clouds':
    st.header('Feature engineering & Emotion-based Word Clouds')

    # Check that the preprocessed dataframe exists in session state
    if 'df_working' not in st.session_state:
        st.warning(
            'No preprocessed dataframe found. Please go to "Upload & Preprocess" and upload/apply preprocessing first.')
    else:
        df = st.session_state['df_working']
        st.write('Dataset loaded from session state.')
        st.write(f'Rows: {len(df):,}')

        # Converting numeric ratings into emotion labels on a 1-5 scale ie. mapping: 1-2 -> negative (anger/sadness), 3 -> neutral, 4-5 -> positive (joy)
        def score_to_emotion(score):
            try:
                s = float(score)
            except Exception:
                return 'neutral'  # return neutral if not numeric or invalid scores
            if s <= 2.0:
                return 'negative'  # equal to or below 2, negative
            elif s == 3.0:
                return 'neutral'  # equal to 3, indifferent
            else:
                return 'positive'  # 4 and 5, positive


        if 'Emotion' not in df.columns:  # Add an Emotion column if not present
            df['Emotion'] = df['Score'].apply(score_to_emotion)  # maps scores to emotions
            st.session_state['df_working'] = df  # update stored df

        st.write(
            'Emotion label distribution:')  # shows distribution of emotion categories, this is to help verify or check the results
        st.write(df['Emotion'].value_counts())

        # TF-IDF vectorizer settings (user can change or make tweaks)
        st.subheader('TF-IDF settings')
        max_features = st.number_input('max_features (TF-IDF)', min_value=1000, max_value=50000, value=10000,
                                       step=1000)  # controls how many features to keep

        # FIXED: Proper string formatting for selectbox
        ngram_range = st.selectbox(
            'ngram_range - Please Choose:',
            options=[(1, 1), (1, 2)],
            format_func=lambda x: "UNIGRAMS (1,1) - Single words (e.g., 'good', 'bad')" if x == (1,1) else "BIGRAMS (1,2) - Word pairs (e.g., 'very good', 'bad service')",
            index=0
        )
        min_df = st.number_input('min_df (int)', min_value=1, max_value=10, value=2)

        if st.button('Run TF-IDF & generate word clouds'):
            texts = df['Text'].astype(
                str).tolist()  # Fit TF-IDF on the Text column, converts text column to list of string
            tfidf = TfidfVectorizer(max_features=int(max_features), ngram_range=ngram_range, min_df=int(min_df),
                                    stop_words='english')

            with st.spinner('Fitting TF-IDF...'): #shows when it is loading
                X = tfidf.fit_transform(
                    texts)  # Learns vocabulary and computes TF-IDF weights and returns memory efficient sparse matrix

            st.success('TF-IDF fitted.')
            st.session_state['tfidf_vectorizer'] = tfidf  # store vectorizer for next page (modeling)

            # For each emotion, compute the mean TF-IDF score across documents and create a word cloud
            emotions = df['Emotion'].unique().tolist()
            emotion_order = ['negative', 'neutral', 'positive']
            emotions = [emo for emo in emotion_order if emo in emotions]
            st.write(f'Generating word clouds for emotions: {emotions}')

            # FIXED: Create columns layout for up to 3 emotions
            if len(emotions) <= 3:
                cols = st.columns(len(emotions))
            else:
                cols = None
            for i, emo in enumerate(emotions):
                with st.spinner(f'Processing {emo}...'):
                    mask = df['Emotion'] == emo
                    if mask.sum() == 0:
                        st.write(f'No documents found for {emo} emotion')
                        continue

                    X_sub = X[mask.values] # Get TF-IDF vectors for this emotion

                    # Compute mean TF-IDF per term
                    mean_tfidf = np.asarray(X_sub.mean(axis=0)).ravel()
                    terms = tfidf.get_feature_names_out()
                    tfidf_scores = dict(zip(terms, mean_tfidf))

                    # Create WordCloud
                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    )
                    wc.generate_from_frequencies(tfidf_scores)

                    # FIXED: Proper visualization with titles
                    if cols and i < len(cols):
                        # Column layout for 3 or fewer emotions
                        with cols[i]:
                            st.subheader(f'{emo.capitalize()} Sentiment')
                            fig, ax = plt.subplots(figsize=(4, 2))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{emo.capitalize()} Words', fontsize=16, pad=20)
                            st.pyplot(fig)
                            plt.close(fig)  # FIXED: Close figure to prevent memory issues
                    else:
                        # Vertical layout for more than 3 emotions
                        st.subheader(f'{emo.capitalize()} Sentiment')
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{emo.capitalize()} Words', fontsize=18, pad=20)
                        st.pyplot(fig)
                        plt.close(fig)  # FIXED: Close figure to prevent memory issues

            st.success('Word clouds generated and displayed above.')

#Modeling & Evaluation Page (Page 4)
if page == 'Modeling & Evaluation':
    st.header('Train models and evaluate')

    if 'df_working' not in st.session_state:
        st.warning('No preprocessed dataframe found. Please go to "Upload & Preprocess" and upload/apply preprocessing first.') #displays so that the user does not skip and step
    elif 'tfidf_vectorizer' not in st.session_state:
        st.warning('No TF-IDF vectorizer found. Please run TF-IDF on the "Features & Word Clouds" page first.')
    else:
        df = st.session_state['df_working'] #retrieves dataset
        tfidf = st.session_state['tfidf_vectorizer'] # retrieves TF-IDF vectorizer

        texts = df['Text'].astype(str).tolist()
        y = df['Emotion'].astype(str).tolist()
        X = tfidf.transform(texts)  # use transform so training is consistent with the vectorizer

        le = LabelEncoder() # Convert labels to numeric for classifiers, imported at the begining
        y_enc = le.fit_transform(y) #categorical emotions(positive, neutral and negative) converted to numeric (0,1,2)

        # Split into train/test
        test_size = st.slider('Test set fraction', 0.1, 0.5, 0.2) # user is allowed to adjust features for test and train with the slider
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)
        st.write(f'Training rows: {X_train.shape[0]:,}, Test rows: {X_test.shape[0]:,}')

        # Model hyperparameters (simple to keep runtime reasonable)
        st.subheader('Model hyperparameters')
        max_iter = st.number_input('LogisticRegression max_iter', value=200, min_value=50, max_value=2000, step=50) #more thorough but slower training
        rf_estimators = st.number_input('RandomForest n_estimators', value=100, min_value=10, max_value=1000, step=10) # better performance but slower training

        col1, col2 = st.columns(2) # Buttons to train models
        with col1:
            if st.button('Train Logistic Regression'):
                with st.spinner('Training Logistic Regression...'): #shows in a form of loading on the page
                    lr = LogisticRegression(max_iter=int(max_iter))
                    lr.fit(X_train, y_train)
                st.success('Logistic Regression trained.') #displays after model has been trained successfully
                joblib.dump(lr, 'logistic_model.joblib') # saves to joblib for external use
                st.session_state['lr_model'] = lr # stores in session state for instant evaluation

        with col2:  # same thing as that of the logistic regression above
            if st.button('Train Random Forest'):
                with st.spinner('Training Random Forest...'):
                    rf = RandomForestClassifier(n_estimators=int(rf_estimators), n_jobs=-1)
                    rf.fit(X_train, y_train)
                st.success('Random Forest trained.')
                joblib.dump(rf, 'rf_model.joblib')
                st.session_state['rf_model'] = rf

        # If models are in session state, evaluate them
        def evaluate_model(name, model, X_test, y_test):
            """Compute metrics and show plots for a trained model."""
            y_pred = model.predict(X_test) #gets models prediction on held out test data
            # For ROC-AUC we need probability estimates (or decision_function)
            try:
                y_score = model.predict_proba(X_test)
                if y_score.shape[1] > 2: # if multiclass, compute macro ROC-AUC
                    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovo', average='macro') #use this for 2 class problems
                else:
                    roc_auc = roc_auc_score(y_test, y_score[:,1])
            except Exception: #if probabilities are unavailable try decision function scores below
                try:
                    y_score = model.decision_function(X_test)
                    roc_auc = roc_auc_score(y_test, y_score)
                except Exception: #if not then reproduce nan
                    roc_auc = float('nan')

            # Macro averaging is a method for computing evaluation metrics across multiple classes in classification problems. It treats all classes equally, regardless of how many samples each class has.
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0) #treating all classes equally, which classes were actually positive
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0) # which classes were correctly identified
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) #balanced mean of precision and recall above, (ideal)
            #display results
            st.subheader(f'Evaluation for {name}')
            st.write('Precision (macro):', f'{prec:.4f}')
            st.write('Recall (macro):', f'{rec:.4f}')
            st.write('F1-score (macro):', f'{f1:.4f}')
            st.write('ROC-AUC:', roc_auc)

            st.write('Classification report:')
            st.text(classification_report(y_test, y_pred, target_names=le.classes_))

            # Confusion matrix visualization, shows actual vs predicted classifications
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            im = ax.imshow(cm, interpolation='nearest')
            ax.set_title('Confusion matrix')
            ax.set_xticks(np.arange(len(le.classes_)))
            ax.set_yticks(np.arange(len(le.classes_)))
            ax.set_xticklabels(le.classes_)
            ax.set_yticklabels(le.classes_)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
            st.pyplot(fig)

            # If we have probabilities and binary case, plot ROC curve for each class
            try:
                if y_score is not None:
                    if y_score.shape[1] == 2: # If multiclass: skip detailed ROC curve plotting here
                        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
                        fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
                        ax2.plot(fpr, tpr)
                        ax2.set_title('ROC curve')
                        ax2.set_xlabel('False Positive Rate')
                        ax2.set_ylabel('True Positive Rate')
                        st.pyplot(fig2)
            except Exception:
                pass

        # Evaluate Logistic regression that has been trained in the session state
        if 'lr_model' in st.session_state:
            evaluate_model('Logistic Regression', st.session_state['lr_model'], X_test, y_test)

        # Evaluate Random Forest that has been trained in the session state
        if 'rf_model' in st.session_state:
            evaluate_model('Random Forest', st.session_state['rf_model'], X_test, y_test)

        # Quick model comparison
        if 'lr_model' in st.session_state and 'rf_model' in st.session_state:
            st.subheader('Quick model comparison (F1 macro)')
            lr = st.session_state['lr_model']
            rf = st.session_state['rf_model']
            lr_pred = lr.predict(X_test)
            rf_pred = rf.predict(X_test)
            lr_f1 = f1_score(y_test, lr_pred, average='macro', zero_division=0) #uses f1 score for comparision
            rf_f1 = f1_score(y_test, rf_pred, average='macro', zero_division=0)
            st.write(f'Logistic Regression F1 (macro): {lr_f1:.4f}')
            st.write(f'Random Forest F1 (macro): {rf_f1:.4f}')
            better = 'Logistic Regression' if lr_f1 > rf_f1 else 'Random Forest' if rf_f1 > lr_f1 else 'Tie'
            st.info(f'Better model by F1 (macro): {better}') #prints out which one was better

# Manual Sentiment Test Page (Page 4)
if page == 'Manual Sentiment Test':
    st.header('Manual Sentiment & Document Analysis')
    if 'tfidf_vectorizer' not in st.session_state:
        st.warning('Please run TF-IDF first.')
    elif 'lr_model' not in st.session_state and 'rf_model' not in st.session_state:
        st.warning('Please train at least one model first.') #allows that the model has been preprocessed and trained and stored in the session state first
    else:
        tfidf = st.session_state['tfidf_vectorizer']
        model_choice = None
        if 'lr_model' in st.session_state and 'rf_model' in st.session_state:
            model_choice = st.selectbox("Select model:", ["Logistic Regression", "Random Forest"]) #if the above is satisfied, allow the user to make a choice
        elif 'lr_model' in st.session_state:
            model_choice = "Logistic Regression"
        elif 'rf_model' in st.session_state:
            model_choice = "Random Forest"
        model = st.session_state['lr_model'] if model_choice == "Logistic Regression" else st.session_state['rf_model']

        le = LabelEncoder() #setting up label encoder already imported above
        le.fit(st.session_state['df_working']['Emotion'].astype(str))

        st.subheader("Input text manually") # manual text entry for analysis
        user_text = st.text_area("Please Enter text:", height=150)
        if st.button("Analyze Typed Text"):
            if not user_text.strip():  #checks foe empty or white spaces
                st.error("Please enter some text.")
            else:
                X_input = tfidf.transform([user_text])
                prediction = model.predict(X_input)[0]
                st.success(f"Predicted Sentiment: **{le.inverse_transform([prediction])[0]}**") #display sentiment

        st.subheader("Upload a file (.txt, .pdf, .csv)") # setting up for a document to be inserted
        st.text_area('NB:CSV must have a Text column,you are advised to rename your CSV column with the reviews as Text')
        uploaded_file = st.file_uploader("Upload file", type=['txt', 'csv']) #only these there can be put, nno unsupported file allowed

        if uploaded_file is not None:
            file_text = ""
            if uploaded_file.type == "text/plain": #direct text extraction for.txt files
                file_text = uploaded_file.read().decode("utf-8", errors="ignore") #(UTF-8) converts text to this format,(errors="ignnore")skipps errors instead of crashinng
            elif uploaded_file.type == "text/csv": #direct text extraction for.csv files
                df_uploaded = pd.read_csv(uploaded_file)
                if 'Text' in df_uploaded.columns: #the leading cell of the file should be named "Text" for it to work
                    file_text = "\n".join(df_uploaded['Text'].dropna().astype(str).tolist())
                else:
                    st.error("CSV must have a 'Text' column.")

            if file_text.strip():
                lines = [line.strip() for line in file_text.split("\n") if line.strip()]
                X_input = tfidf.transform(lines)
                predictions = model.predict(X_input)
                sentiments = le.inverse_transform(predictions)
                for i, (line, sent) in enumerate(zip(lines, sentiments)):
                    if i < 10:
                        st.write(f"**Text:** {line}")
                        st.write(f"Predicted Sentiment: **{sent}**")
                        st.write("---")
                st.success(f"Document processed. Segments analyzed: {len(lines)}")
            else:
                st.error("No readable text found.")

#Deployment & Notes Page (Page 5)
if page == 'Deployment & Notes':
    st.header('Deploying the app and Information on Project Setup')

    st.write('This section contains practical deployment instructions, how to set up the project and also how to reproduce the saved model.')

    st.subheader('Requirements')
    st.write('You will typically need to install:')
    st.code('\n'.join([
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'wordcloud',
        'joblib'
        'kagglehub'
    ]))

    st.subheader('Project Setup')
    st.write('- download the dataset from kaggle using this link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/')
    st.write('- create a virtual environment for the project using this command python -m venv <environment_name>')
    st.write('- install the packages using the requirements.txt file using the pip install -r requirements.tx')
    st.write('- to run the application, run streamlit run TextAmazon.py')

    st.subheader('Notes on reproducibility')
    st.write('Save fitted TF-IDF and trained models using joblib and commit small example datasets to the repo for testing.')

    st.success('End of app notes. Navigate back using the sidebar.')

    #Helpful tip printed to the app (not a title on each page)
    st.caption('Streamlit app: Emotion WordClouds — TF-IDF + Logistic Regression & Random Forest')

st.sidebar.markdown("___")
st.sidebar.markdown("**PROJECT TEAM**")
st.sidebar.markdown("Raymond Tetteh (22255065)")
st.sidebar.markdown("Rebecca Boakyewaa Anderson (11410707)")
st.sidebar.markdown("Rene Dwamena (11410590)")
st.sidebar.markdown("Rowland Walker Mensah (22252448)")
st.sidebar.markdown("Ricky Nelson Adzakpa (11410541)")

