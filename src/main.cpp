#include <iostream>
#include <Eigen/Dense>
#include <sqlite3.h>


#define DEBUG_SHAPE(x) std::cout<< "("<< x.rows()<<","<<x.cols()<<")"<< std::endl
#define DEBUG_PRINT(x) std::cout << "(" << x << ")" << std::endl

//Activation function struct
typedef Eigen::MatrixXf (*actfn_ptr)(Eigen::MatrixXf &m1);
struct Afunc
{
    actfn_ptr act_fn;
    actfn_ptr act_dfn;
};


int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 25;//<--- size of a single datapoint
    int input_layer_size = 10;//<--- Number of training poìnts

    //first layer 
    int first_layer_size = 10;//<--- Size of the first layer

    actfn_ptr sigmoid   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(1.0f/(1.0f + m.array().exp()));};
    actfn_ptr sigmoid_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};

    Afunc activ = Afunc{
                    sigmoid,
                    sigmoid_d
                };


    Eigen::MatrixXf x = Eigen::MatrixXf::Random(input_layer_size,input_vector_size);

    Eigen::MatrixXf m = Eigen::MatrixXf::Random(first_layer_size,input_vector_size);
    Eigen::VectorXf b = Eigen::VectorXf::Random(first_layer_size);

    Eigen::MatrixXf l1 = (m * x.transpose()).colwise() + b;
    DEBUG_PRINT(activ.act_dfn(l1));
    DEBUG_PRINT(activ.act_fn(l1));

    return 0;
}