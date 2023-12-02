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
struct Layer{

    int in_size;
    int l_size;
    Afunc a_fn;
    Eigen::MatrixXf w;
    Eigen::VectorXf b;

    Layer(int input_size, int layer_size, Afunc activation_function)
        :in_size(input_size),
        l_size(layer_size),
        a_fn(activation_function),
        w(Eigen::MatrixXf::Random(layer_size,input_size) * 0.5f),
        b(Eigen::VectorXf::Random(layer_size) *0.5f){
    }

    Eigen::MatrixXf ForwardPropagate(const Eigen::MatrixXf& input){
        return a_fn.act_fn((w * input.transpose()).colwise() + b);
    }
    void BackPropagate(){

    }
};
int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 25;//<--- size of a single datapoint
    int input_layer_size = 10;//<--- Number of training poìnts

    //first layer 
    int first_layer_size = 10;//<--- Size of the first layer
    int second_layer_size = 20;//<--- Size of the first layer


    actfn_ptr sigmoid   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(1.0f/(1.0f + m.array().exp()));};
    actfn_ptr sigmoid_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};

    actfn_ptr relu   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseMax(0));};
    //actfn_ptr relu_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};
    Afunc activ_sigmoid = Afunc{
                    sigmoid,
                    sigmoid_d
                };

    //Input 
    Eigen::MatrixXf x = Eigen::MatrixXf::Random(input_layer_size,input_vector_size);

    Eigen::MatrixXf m = Eigen::MatrixXf::Random(first_layer_size,input_vector_size) * 0.5f;
    Eigen::VectorXf b = Eigen::VectorXf::Random(first_layer_size) *0.5f;
    Eigen::MatrixXf l1 = (m * x.transpose()).colwise() + b;


    Layer l1_cls(input_vector_size,input_layer_size,activ_sigmoid);
    DEBUG_PRINT(activ_sigmoid.act_dfn(l1));
    DEBUG_PRINT(activ_sigmoid.act_dfn(l1));
    DEBUG_PRINT(activ_sigmoid.act_fn(l1));
    DEBUG_PRINT(l1_cls.ForwardPropagate(x));

    return 0;
}