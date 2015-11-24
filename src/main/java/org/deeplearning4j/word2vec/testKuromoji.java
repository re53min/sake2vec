package org.deeplearning4j.word2vec;

/**
 * Created by b1012059 on 2015/11/09.
 */

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import java.util.List;

public class testKuromoji {
    public static void main(String[] args) {
        Tokenizer tokenizer = new Tokenizer() ;
        List<Token> tokens = tokenizer.tokenize("人間の双曲線割引的な報酬系の振る舞いが、人間の意志を決定する。");
        for (Token token : tokens) {
            String[] features = token.getAllFeaturesArray();
            //System.out.print(token.getSurface()+" ");// + "\t" + token.getAllFeatures());
            if(features[0].equals("名詞")) System.out.print(token.getSurface() + " ");
        }
    }
}
