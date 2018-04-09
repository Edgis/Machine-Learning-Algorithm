/**
 * Random-Forest implementation in JAVA
 * @Author EDGIS
 * @Contact guoxianwhu@foxmail.com
 */
package com.rf.real;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
/**
 * Random Forest
 */
public class RandomForest {

    /**
     * 可用的线程数量
     * */
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    /** 
     *target类别数量
     * */
    public static int C;
    /** 
     * 属性（列）的数量
     * */
    public static int M;
    /** 
     *属性扰动，每次从M个属性中随机选取Ms个属性，Ms = log2(M)
     */
    public static int Ms;
    /** the collection of the forest's decision trees */
    private ArrayList<DTree> trees;
    /** 
     * 开始时间
     * */
    private long time_o;
    /** the number of trees in this random tree */
    private int numTrees;
    /** 
     * 为了实时显示进度，每建立一棵树的更新量
     */
    private double update;
    /**
     * 为了实时显示进度，初试值 
     */
    private double progress;
    /** importance Array  */
    private int[] importances;
    /** key = a record from data matrix
     * value = RF的分类结果*/
    private HashMap<int[],int[]> estimateOOB;
    /** all of the predictions from RF */
    private ArrayList<ArrayList<Integer>> Prediction;
    /** RF的错误率 */
    private double error;
    /** 控制树生长的进程池 */
    private ExecutorService treePool;
    /** 原始训练数据 */
    private ArrayList<int[]> train_data;
    /** 测试数据*/
    private ArrayList<int[]> testdata;
    /**
     * Initializes a Random forest 
     * @param numTrees			RF的数量
     * @param train_data		原始训练数据
     * @param t_data			测试数据
     */
    public RandomForest(int numTrees, ArrayList<int[]> train_data, ArrayList<int[]> t_data ){
        this.numTrees = numTrees;
        this.train_data = train_data;
        this.testdata = t_data;
        trees = new ArrayList<DTree>(numTrees);
        update = 100 / ((double)numTrees);
        progress = 0;
        StartTimer();
        System.out.println("creating "+numTrees+" trees in a random Forest. . .");
        System.out.println("total data size is "+train_data.size());
        System.out.println("number of attributes " + (train_data.get(0).length-1));
        System.out.println("number of selected attributes " + ((int)Math.round(Math.log(train_data.get(0).length-1)/Math.log(2) + 1)));
        estimateOOB = new HashMap<int[],int[]>(train_data.size());
        Prediction = new ArrayList<ArrayList<Integer>>();
    }
    /**
     * Begins the creation of random forest 
     */
    public void Start() {
        System.out.println("Num of threads started : " + NUM_THREADS);
        System.out.println("Running...");
        treePool = Executors.newFixedThreadPool(NUM_THREADS);
        for (int t=0; t < numTrees; t++){
            System.out.println("structing " + t + " Tree");
            treePool.execute(new CreateTree(train_data,this,t+1));
            //System.out.print(".");
        }
        treePool.shutdown();
        try {
            treePool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
        } catch (InterruptedException ignored){
            System.out.println("interrupted exception in Random Forests");
        }
        System.out.println("");
        System.out.println("Finished tree construction");
        TestForest(trees, testdata);
        CalcImportances();
        System.out.println("Done in "+TimeElapsed(time_o));
    }

    /**
     * @param collec_tree   the collection of the forest's decision trees
     * @param test_data	    测试数据集
     */
    private void TestForest(ArrayList<DTree> collec_tree, ArrayList<int[]> test_data ) {
        int correstness = 0 ;
        int k = 0;
        ArrayList<Integer> actualLabel = new ArrayList<Integer>();
        for(int[] rec:test_data){
            actualLabel.add(rec[rec.length-1]);
        }
        int treeNumber = 1;
        for(DTree dt:collec_tree){
            dt.CalculateClasses(test_data, treeNumber);
            Prediction.add(dt.predictions);
            treeNumber++;
        }
        for(int i = 0; i<test_data.size(); i++){
            ArrayList<Integer> Val = new ArrayList<Integer>();
            for(int j =0; j<collec_tree.size(); j++){
                Val.add(Prediction.get(j).get(i));//The collection of each Tree's prediction in i-th record
            }
            int pred = labelVote(Val);//Voting algorithm
            if(pred == actualLabel.get(i)){
                correstness++;
            }
        }
        System.out.println("Accuracy of Forest is : " + (100 * correstness / test_data.size()) + "%");
    }

    /**
     * Voting algorithm
     * @param treePredict   The collection of each Tree's prediction in i-th record
     */
    private int labelVote(ArrayList<Integer> treePredict){
        // TODO Auto-generated method stub
        int max=0, maxclass=-1;
        for(int i=0; i<treePredict.size(); i++){
            int count = 0;
            for(int j=0; j<treePredict.size(); j++){
                if(treePredict.get(j) == treePredict.get(i)){
                    count++;
                }
                if(count > max){
                    maxclass = treePredict.get(i);
                    max = count;
                }
            }
        }
        return maxclass;
    }
    /**
     * 计算RF的分类错误率
     */
    private void CalcErrorRate(){
        double N=0;
        int correct=0;
        for (int[] record:estimateOOB.keySet()){
            N++;
            int[] map=estimateOOB.get(record);
            int Class=FindMaxIndex(map);
            if (Class == DTree.GetClass(record))
                correct++;
        }
        error=1-correct/N;
        System.out.println("correctly mapped "+correct);
        System.out.println("Forest error rate % is: "+(error*100));
    }
    /**
     * 更新  OOBEstimate
     * @param record	        a record from data matrix
     * @param Class		
     */
    public void UpdateOOBEstimate(int[] record, int Class){
        if (estimateOOB.get(record) == null){
            int[] map = new int[C];
            //System.out.println("class of record : "+Class);map[Class-1]++;
            estimateOOB.put(record,map);
        }
        else {
            int[] map = estimateOOB.get(record);
            map[Class-1]++;
        }
    }
    /**
     * calculates the importance levels for all attributes.
     */
    private void CalcImportances() {
        importances = new int[M];
        for (DTree tree:trees){
            for (int i=0; i<M; i++)
                importances[i] += tree.getImportanceLevel(i);
        }
        for (int i=0;i<M;i++)
            importances[i] /= numTrees;
        System.out.println("The forest-wide importance as follows:");
        for (int j=0; j<importances.length; j++){
            System.out.println("Attr" + j + ":" + importances[j]);
        }
    }
    /** 计时开始 */
    private void StartTimer(){
        time_o = System.currentTimeMillis();
    }
    /**
     * 创建一棵决策树
     */
    private class CreateTree implements Runnable{ 
        /** 训练数据 */
        private ArrayList<int[]> train_data;
        /** 随机森林 */
        private RandomForest forest;
        /** the numb of RF */
        private int treenum;
   
        public CreateTree(ArrayList<int[]> train_data, RandomForest forest, int num){
            this.train_data = train_data;
            this.forest = forest;
            this.treenum = num;
        }
        /**
         * Create the decision tree
         */
        public void run() {
            System.out.println("Creating a Dtree num : " + treenum + " ");
            trees.add(new DTree(train_data, forest, treenum));
            //System.out.println("tree added in RandomForest.AddTree.run()");
            progress += update;
            System.out.println("---progress:" + progress);
        }
    }

    /**
     * Evaluates testdata
     * @param record	a record to be evaluated
     */
    public int Evaluate(int[] record){
        int[] counts=new int[C];
        for (int t=0;t<numTrees;t++){
            int Class=(trees.get(t)).Evaluate(record);
            counts[Class]++;
        }
        return FindMaxIndex(counts);
    }
    
    public static int FindMaxIndex(int[] arr){
        int index=0;
        int max = Integer.MIN_VALUE;
        for (int i=0;i<arr.length;i++){
            if (arr[i] > max){
                max=arr[i];
                index=i;
            }
        }
        return index;
    }
    

    /**
     * @param timeinms	        开始时间
     * @return			the hr,min,s 
     */
    private static String TimeElapsed(long timeinms){
        int s=(int)(System.currentTimeMillis()-timeinms)/1000;
        int h=(int)Math.floor(s/((double)3600));
        s-=(h*3600);
        int m=(int)Math.floor(s/((double)60));
        s-=(m*60);
        return ""+h+"hr "+m+"m "+s+"s";
    }
}
