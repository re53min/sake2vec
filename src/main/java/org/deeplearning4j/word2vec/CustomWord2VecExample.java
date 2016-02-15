package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
/**
 * Created by agibsonccc on 10/9/14.
 */

public class CustomWord2VecExample {
    private static Logger log = LoggerFactory.getLogger(CustomWord2VecExample.class);


    public static void main(String[] args) throws Exception {

        String filePath = new ClassPathResource("corpus_v2.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache();
        WeightLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(50)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        // Customizing params
        int batchSize = 1000;
        int iterations = 30;
        int layerSize = 30;

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(batchSize)
                .minWordFrequency(2)
                .iterations(iterations)
                .layerSize(layerSize)
                .sampling(1e-2)
                .windowSize(15)
                .negativeSample(15)
                .minLearningRate(1e-2)
                .lookupTable(table)
                .vocabCache(cache)
                .seed(42)
                .iterate(iter)
                .tokenizerFactory(t)
        .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        double[] sim = new double[10];
        sim[0] = vec.similarity("獺祭", "薫酒");
        sim[1] = vec.similarity("久保田", "薫酒");
        sim[2] = vec.similarity("八海山", "薫酒");
        sim[3] = vec.similarity("黒龍", "薫酒");
        sim[4] = vec.similarity("飛露喜", "薫酒");
        sim[5] = vec.similarity("田酒", "薫酒");
        sim[6] = vec.similarity("天狗舞", "薫酒");
        sim[7] = vec.similarity("蓬莱泉", "薫酒");
        sim[8] = vec.similarity("出羽桜", "薫酒");
        sim[9] = vec.similarity("〆張鶴", "薫酒");

        for(int i = 0; i < sim.length; i++) System.out.println(sim[i]);

        log.info("Save vectors....");
        WordVectorSerializer.writeWordVectors(vec, "corpus_v2_Model.txt");

        log.info("Writing word vectors to text file....");
        // Write word
        WordVectorSerializer.writeFullModel(vec, "corpus_v2_FullModel.txt");

        /*log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println(lst);

        log.info("Word Formula");
        Collection<String> kingList = vec.wordsNearest(Arrays.asList("week", "season"), Arrays.asList("years"), 10);
        System.out.println(kingList);*/
    }

}
