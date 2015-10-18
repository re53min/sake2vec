package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecExample {

    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);

    public static void main(String[] args) throws Exception {

        Nd4j.getRandom().setSeed(133);

        ClassPathResource resource = new ClassPathResource("日本酒コーパス.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });


        int batchSize = 1000;
        int iterations = 30;
        int layerSize = 30;
        Word2Vec vec;
        File file = new File("日本酒-30.txt");
        List<String> posi = new ArrayList();
        List<String> nega = new ArrayList();


        if (file.exists()) {
            System.out.println("モデルがありました。再利用します");
            vec = WordVectorSerializer.loadGoogleModel(file, true);

            //testing similarity
            double sim = vec.similarity("獺祭", "八海山");
            log.info("Similarity between A and B: " + sim);

            //testing wordsNearest
            posi.add("獺祭");
            nega.add("甘い");
            nega.add("辛い");
            Collection<String> similar = vec.wordsNearest(posi, nega, 20);
            log.info(String.valueOf(similar));

        } else {
            System.out.println("モデルがありません。学習を始めます");
            vec = new Word2Vec.Builder()
                    .batchSize(batchSize) //# words per minibatch.
                    .sampling(1e-5) // negative sampling. drops words out
                    .minWordFrequency(5) //
                    .useAdaGrad(false) //
                    .layerSize(layerSize) // word feature vector size
                    .iterations(iterations) // # iterations to train
                    .learningRate(0.025) //
                    .minLearningRate(1e-2) // learning rate decays wrt # words. floor learning
                    .negativeSample(10) // sample size 10 words
                    .iterate(iter) //
                    .tokenizerFactory(t)
                    .build();
            vec.fit();

            InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
            table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

            //save word vector
            System.out.println("モデルを保存します");
            WordVectorSerializer.writeWordVectors(vec, "日本酒-30.txt");

            //testing similarity
            double sim = vec.similarity("獺祭", "八海山");
            log.info("Similarity between A and B: " + sim);

            //testing wordsNearest
            posi.add("獺祭");
            nega.add("甘い");
            nega.add("辛い");
            Collection<String> similar = vec.wordsNearest(posi, nega, 20);
            log.info(String.valueOf(similar));



        }

    }
}