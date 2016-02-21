package org.deeplearning4j.nnpractice;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by b1012059 on 2016/02/04.
 */
public class NLP {
    //private static Logger log = LoggerFactory.getLogger(NLP.class);
    private Tokenizer tokenizer;
    private List<Token> tokens;
    private HashMap<String, Integer> wordCount;
    private HashMap<String, Integer> wordToId;
    private ArrayList<String> ret;

    public NLP(String text){
        this.tokenizer = new Tokenizer();
        this.ret = new ArrayList<>();
        this.wordCount = new HashMap<>();
        this.wordToId = new HashMap<>();

        if(text != null){
            tokens = tokenizer.tokenize(text);
            createVector();
        }

    }

    private void createVector(){
        int id = 0;

        //log.info("Stating Morphological Analysis");
        tokens.forEach(token -> {
            String[] features = token.getAllFeaturesArray();

            ret.add(token.getSurface());
            int counter = 1;

            /*
            単語がすでにリストにある場合のみカウントを増やす
            初登場だったら新規に登録
             */
            if(wordCount.containsKey(token.getSurface())){
                counter = wordCount.get(token.getSurface()) + 1;
            }
            wordCount.put(token.getSurface(), counter);
        });

        //log.info("Creating WordToId");
        for(String key : wordCount.keySet()){
            //単語and freqency
            wordToId.put(key, id);
            id++;
        }
    }

    public Map<String, Integer> createNgram(int n) {
        // 生成されるn-gramの数（ループ回数になる）
        int numberOfNgram = ret.size() - n + 1;
        Map<String, Integer> ngramMap = new HashMap<>();
        StringBuilder ngramSb = new StringBuilder();

        // n-gramとその出現回数を格納したMapを生成
        //log.info("Creating N-gram");
        for(int i = 0; i < numberOfNgram; i++) {
            // ngramを1つ生成
            for (int j = i; j < i + n; j++) {
                ngramSb.append(ret.get(j)).append(" ");
            }
            ngramSb.deleteCharAt(ngramSb.length() - 1);
            String ngramStr = ngramSb.toString();
            ngramSb.delete(0, ngramSb.length());

            // 生成したn-gramをMapに入れてカウント
            if (ngramMap.containsKey(ngramStr)) {
                ngramMap.put(ngramStr, ngramMap.get(ngramStr) + 1);
            } else {
                ngramMap.put(ngramStr, 1);
            }
        }

        return ngramMap;
    }

    public ArrayList<String> getRet(){
        return this.ret;
    }

    public HashMap<String, Integer> getWordCount(){
        return this.wordCount;
    }

    public HashMap<String, Integer> getWordToId(){
        return this.wordToId;
    }
}
