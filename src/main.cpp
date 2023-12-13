#include <iostream>
#include <Eigen/Dense>
#include <sqlite3.h>


#define DEBUG_SHAPE(x) std::cout<< "("<< x.rows()<<","<<x.cols()<<")"<<"("<< x.size()<<")"<< std::endl
#define DEBUG_SHOW_MATRIX(x) std::cout << "(" << x << ")" << "("<< x.rows()<<","<<x.cols()<<")"<<"("<< x.size()<<")"<< std::endl
#define DEBUG_PRINT(x) std::cout << x << std::endl

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
        Eigen::MatrixXf it = (w * input).colwise() + b;
        return a_fn.act_fn(it);
    }
    Eigen::MatrixXf BackPropagate(const Eigen::MatrixXf& input, const Eigen::MatrixXf& a,const Eigen::MatrixXf& a_prev, int dclass_size){
        Eigen::MatrixXf dz = a-input;
        Eigen::MatrixXf dw = (dz  * a_prev.transpose()/dclass_size);
        Eigen::MatrixXf db = (dz.sum()/dclass_size);

    }
};
Eigen::MatrixXf OneHot( const Eigen::VectorXi& labels ){
    Eigen::MatrixXf oneh = Eigen::MatrixXf::Zero(labels.size(),labels.maxCoeff()+1);
    //Ctm que feo esto
    for(int i=0;i<labels.size();i++){
        oneh(i,labels(i)) = 1;
    }
    return oneh.transpose();
}
int main(int argc, char *argv[]){

    //Input layer 
    int input_vector_size = 5;//<--- size of a single datapoint
    int input_layer_size = 10;//<--- Number of training poìnts

    Eigen::VectorXi label_data(input_layer_size);
    label_data <<  1,3,4,3,5,4,2,1,4,2;

    //Layer sizes
    int first_layer_size = 10;//<--- Size of the first layer
    int second_layer_size = 20;//<--- Size of the first layer
    int output_layer_size = label_data.maxCoeff()+1;//<---last layer output must coincide with data classes

    actfn_ptr softmax   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf( m.array().exp() / m.array().exp().sum());};
  //  actfn_ptr softmax_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.exp()/m.exp().sum());};

    actfn_ptr sigmoid   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(1.0f/(1.0f + m.array().exp()));};
    actfn_ptr sigmoid_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.array() * ((1.0f - m.array())));};

    actfn_ptr relu   =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseMax(0));};
    actfn_ptr relu_d =  *[]( Eigen::MatrixXf &m) {return Eigen::MatrixXf(m.cwiseGreater(0).cast<float>());};

    Afunc activ_softmax = Afunc{
                    softmax,
                    softmax
                };

    Afunc activ_sigmoid = Afunc{
                    sigmoid,
                    sigmoid_d
                };
    Afunc activ_relu = Afunc{
                    relu,
                    relu_d
                };

    Eigen::MatrixXf x = Eigen::MatrixXf::Random(input_layer_size,input_vector_size);

    Layer l1_cls(input_vector_size,first_layer_size,activ_relu);
    Layer l2_cls(first_layer_size,second_layer_size,activ_sigmoid);
    Layer l3_cls(second_layer_size,output_layer_size,activ_softmax);

    Eigen::MatrixXf m1 = l1_cls.ForwardPropagate(x.transpose());
    Eigen::MatrixXf m2 = l2_cls.ForwardPropagate(m1);
    Eigen::MatrixXf m3 = l3_cls.ForwardPropagate(m2);


    Eigen::MatrixXf onehot = OneHot(label_data);

    DEBUG_PRINT(input_layer_size);
    DEBUG_SHOW_MATRIX(m1);
    DEBUG_SHOW_MATRIX(m2);
    DEBUG_SHOW_MATRIX(m3);
    DEBUG_SHOW_MATRIX(onehot);
    DEBUG_SHOW_MATRIX((m3-onehot));

    return 0;
}