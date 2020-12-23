# IMPORT PACKAGES
## WORD EMBEDDINGS
import gensim
### source: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
## DATA MANIPULATION
import pandas as pd

# Trains a DBOW architecture from a supplied corpus
# Input: trainDf - dataframe - dataframe that contains the training radiology reports
#        loadModel - string - load a previously saved model. requires the path to model.
#        idColumn - string - indicates the column that contains the ids for each report
#        vectorSize - integer - size of the document vector
#        epochs - integer - # of epochs
#        seed - integer - controls for randomness
#        dbWords - integer - either 0 or 1, if 1 then skip-gram is trains the word embeddings first prior to dbow. 
#                               with 0 word embeddings are randomly initialized.
#        save - integer - indicate if you want save the model, Default 0 - False. 1 - True
# Output: model - gensim object - trained DBOW architecture
def dbowArchitectureBuilder(trainDf,
                            savePath="",
                            loadModel="",
                            idColumn="ROW_ID",
                            textColumn="TEXT",
                            vectorSize = 600,
                            epochs = 500,
                            seed = 2,
                            dbWords = 1):
    
    if loadModel!="":
        model = gensim.models.doc2vec.Doc2Vec.load(loadModel)
    else:
        ## CREATE TAG DOCUMENT OBJECT
        trainDocs = []
        for i in range(0,len(trainDf)):
            row = trainDf.iloc[i,:]
            imageid = row[idColumn]
            doc = row[textColumn]
            T = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [imageid])
            trainDocs.append(T)
        ## DBOW
        model = gensim.models.doc2vec.Doc2Vec(dm=0,
                                            vector_size=vectorSize,
                                            negative=5,
                                            seed=seed,
                                            hs=0,
                                            min_count=2,
                                            alpha=0.05,
                                            epochs=epochs,
                                            dbow_words=dbWords,
                                            workers=2)
        model.build_vocab(trainDocs)
        model.train(trainDocs, total_examples=model.corpus_count, epochs=model.epochs)
    if savePath != "":
        model.save(savePath + "/documentEmbedding.model")
    return model

# using the model from dbowArchitectureBuilder, this function infers the document vector for each report from
#   the training and test set.
# Input: df - dataframe - contains the train and test text
#        idColumn - string - name of the id column of the dataframes
#        textColumn - string - name of the text column of the dataframes
#        model - gensim object - output from dbowArchitectureBuilder()
# Output: list of the document vectors for training and test sets
def documentVectorFeatures(df,
                           model,
                           idColumn = "imageid",
                           textColumn = "text"): 
    # FEATURE EXTRACTION
    ## EMPTY LIST
    docVecList = list()
    ## INFER VECTORS
    for i in range(0,len(df)):
        row = df.iloc[i,:]
        imageid = row[idColumn]
        doc = row[textColumn]
        T = gensim.utils.simple_preprocess(doc)
        docVec = model.infer_vector(T)
        docRow = pd.DataFrame(columns=list(range(0,len(docVec))))
        docRow.loc[i] = docVec
        docRow[idColumn] = imageid
        docVecList.append(docRow)
    docVecTable = pd.concat(docVecList)

    return docVecTable
