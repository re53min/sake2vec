package org.deeplearning4j.word2vec;

/**
 * Created by b1012059 on 2015/04/25.
 */
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Sake2vec {
    String fileName;
    Word2Vec vec;

    public Sake2vec(String fileName) {
        this.fileName = fileName;
    }

    public void Sake2vecExample() throws Exception {
        ClassPathResource resource = new ClassPathResource(this.fileName);
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        //DocumentIterator iter = new FileDocumentIterator(resource.getFile());
        TokenizerFactory t = new DefaultTokenizerFactory();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
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

        int layerSize = 300;

        vec = new Word2Vec.Builder().sampling(1e-5)
                .minWordFrequency(5).batchSize(1000).useAdaGrad(false).layerSize(layerSize)
                .iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();


        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(200).useAdaGrad(false)
                .normalize(false).usePca(false).build();


        vec.lookupTable().plotVocab(tsne);


    }

    public double sake2vecResult() {
        double sim = 0;
        try {
            Sake2vecExample();
            //similarity(string, string):指定した単語の類似値
            sim = vec.similarity("people", "money");
            System.out.println("Similarity between people and money " + sim);

            //wordsNearest(string, int):指定した単語に近い単語をint数表示
            Collection<String> similar = vec.wordsNearest("money", 20);
            System.out.println(similar);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return sim;
    }
}