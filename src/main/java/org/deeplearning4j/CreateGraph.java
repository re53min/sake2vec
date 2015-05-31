package org.deeplearning4j;

import org.jfree.chart.*;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.plot.PlotOrientation;

import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * グラフ作成
 * Created by b1012059 on 2015/05/31.
 */
public class CreateGraph extends JFrame{
    private String graphTitle;
    private String categoryAxisLabel;
    private String valueAxisLabel;

    public CreateGraph(String TITLE, String CATEGORY, String VALUE) {
        this.graphTitle = TITLE;
        this.categoryAxisLabel = CATEGORY;
        this.valueAxisLabel = VALUE;
    }


    public ChartPanel createGraphPanel(int dataCount, ArrayList<Double> resultData){
        File file = new File("./graphTest.jpeg");
        String str;
        ChartFactory.setChartTheme(StandardChartTheme.createLegacyTheme());
        DefaultCategoryDataset data = new DefaultCategoryDataset();

        for(int i = 0; i < dataCount-1; i = i + 20){
            str = Integer.toString(i);
            data.addValue(resultData.get(i), "二乗誤差", str);
        }

        JFreeChart chart = ChartFactory.createLineChart(
                graphTitle,
                categoryAxisLabel,
                valueAxisLabel,
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false);

        try {
            ChartUtilities.saveChartAsJPEG(file, chart, 800, 500);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ChartPanel cpanel = new ChartPanel(chart);
        return cpanel;

        //getContentPane().add(cpanel, BorderLayout.CENTER);
    }
}
