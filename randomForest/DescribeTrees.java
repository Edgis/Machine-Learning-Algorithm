/**
 * Random-Forest implementation in JAVA
 * @Author EDGIS
 * @Contact guoxianwhu@foxmail.com
 */
package com.rf.real;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DescribeTrees {
    //把txt文件作为输入，导入到randomForest中
    BufferedReader br = null;
    String path;
    public DescribeTrees(String path){
        this.path = path;
    }

    public ArrayList<int[]> CreateInput(String path){
        ArrayList<int[]> DataInput = new ArrayList<int[]>();
        try {
            String sCurrentLine;
            br = new BufferedReader(new FileReader(path));

            while ((sCurrentLine = br.readLine()) != null) {
                ArrayList<Integer> spaceIndex = new ArrayList<Integer>();//空格的index
                int i;
                if(sCurrentLine != null){
                    sCurrentLine = " " + sCurrentLine + " ";
                    for(i=0; i < sCurrentLine.length(); i++){
                        if(Character.isWhitespace(sCurrentLine.charAt(i)))
                            spaceIndex.add(i);
                    }
                    int[] DataPoint = new int[spaceIndex.size()-1];
                    for(i=0; i<spaceIndex.size()-1; i++){
                        DataPoint[i]=Integer.parseInt(sCurrentLine.substring(spaceIndex.get(i)+1, spaceIndex.get(i+1)));
                    }
                    /* print DataPoint
                    for(k=0; k<DataPoint.length; k++){
                        //System.out.print("-");
                        System.out.print(DataPoint[k]);
                        System.out.print(" ");

                    }
                    **/
                    DataInput.add(DataPoint);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null)
                    br.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        return DataInput;
    }
}
