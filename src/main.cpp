#include <iostream>
#include <Eigen/Dense>
#include <sqlite3.h>


#define DEBUG_SHAPE(x) std::cout<< "("<< x.rows()<<","<<x.cols()<<")"<< std::endl

int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 25;//<--- size of a single datapoint
    int input_layer_size = 10;//<--- Number of training poìnts

    //first layer 
    int first_layer_size = 10;//<--- Size of the first layer

    Eigen::MatrixXf x = Eigen::MatrixXf::Random(input_layer_size,input_vector_size);

    Eigen::MatrixXf m = Eigen::MatrixXf::Random(first_layer_size,input_vector_size);
    Eigen::VectorXf b = Eigen::MatrixXf::Random(first_layer_size,1);
    DEBUG_SHAPE(x);
    DEBUG_SHAPE(x.transpose());
    DEBUG_SHAPE(m);
    DEBUG_SHAPE(b);
    Eigen::MatrixXf l1 = m * x.transpose();
    DEBUG_SHAPE(l1);

    std::cout << l1 <<std::endl;
    return 0;
}