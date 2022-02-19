#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
template <typename Dtype>
void ALSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}
 
template <typename Dtype>
void ALSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + this->int_to_str(this->T_);
  (*names)[1] = "c_T";
}
 
template <typename Dtype>
void ALSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h";
  (*names)[1] = "mask";
}
 
template <typename Dtype>
void ALSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();
 
  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(1);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
 
  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);
 
  LayerParameter attention_param;
  attention_param.set_type("InnerProduct");
  attention_param.mutable_inner_product_param()->set_num_output(256);
  attention_param.mutable_inner_product_param()->set_bias_term(false);
  attention_param.mutable_inner_product_param()->set_axis(2);
  attention_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
 
  LayerParameter biased_attention_param(attention_param);
  biased_attention_param.mutable_inner_product_param()->set_bias_term(true);
  biased_attention_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler); // weight + bias
 
  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);
 
  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);
 
  LayerParameter softmax_param;
  softmax_param.set_type("Softmax");
  softmax_param.mutable_softmax_param()->set_axis(-1);
 
  LayerParameter split_param;
  split_param.set_type("Split");
 
  LayerParameter scale_param;
  scale_param.set_type("Scale");
 
  LayerParameter permute_param;
  permute_param.set_type("Permute");
 
  LayerParameter reshape_param;
  reshape_param.set_type("Reshape");
 
  LayerParameter bias_layer_param;
  bias_layer_param.set_type("Bias");
 
  LayerParameter pool_param;
  pool_param.set_type("Pooling");
 
  LayerParameter reshape_layer_param;
  reshape_layer_param.set_type("Reshape");
 
  BlobShape input_shape;
  input_shape.add_dim(1);  // c_0 and h_0 are a single timestep
  input_shape.add_dim(this->N_);
  input_shape.add_dim(num_output);
 
  net_param->add_input("c_0");
  net_param->add_input_shape()->CopyFrom(input_shape);
 
  net_param->add_input("h_0");
  net_param->add_input_shape()->CopyFrom(input_shape);
 
  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(1);
 
  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->set_name("x_slice");
  x_slice_param->add_bottom("x");
 
  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
/*
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
  }
  if (this->static_input_) {
    // Add layer to transform x_static to the gate dimension.
    //     W_xc_x_static = W_xc_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xc_x_static");
    x_static_transform_param->add_param()->set_name("W_xc_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xc_x_static");
    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    new_shape->add_dim(this->N_);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->add_bottom("W_xc_x_static");
    reshape_param->add_top("W_xc_x_static");
  }
  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");
*/
 
  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);
 
  LayerParameter output_m_layer;
  output_m_layer.set_name("m_concat");
  output_m_layer.set_type("Concat");
  output_m_layer.add_top("mask");
  output_m_layer.mutable_concat_param()->set_axis(0); // out put 2
 
  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = this->int_to_str(t - 1);
    string ts = this->int_to_str(t);
 
    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("x_" + ts);
 
    // Add a layer to permute x
    {
      LayerParameter* permute_x_param = net_param->add_layer();
      permute_x_param->CopyFrom(permute_param);
      permute_x_param->set_name("permute_x_" + ts);
      permute_x_param->mutable_permute_param()->add_order(2);
      permute_x_param->mutable_permute_param()->add_order(0);
      permute_x_param->mutable_permute_param()->add_order(1);
      permute_x_param->mutable_permute_param()->add_order(3);
      permute_x_param->add_bottom("x_" + ts);
      permute_x_param->add_top("x_p_" + ts);
    }
    //
     
 
    // Add a layer to generate attention weights
    {
      LayerParameter* att_m_param = net_param->add_layer();
      att_m_param->CopyFrom(biased_attention_param);
      att_m_param->set_name("att_m_" + tm1s);
      att_m_param->add_bottom("h_" + tm1s);
      att_m_param->add_top("m_" + tm1s);     //     }
   {
      LayerParameter* permute_x_a_param = net_param->add_layer();
      permute_x_a_param->CopyFrom(permute_param);
      permute_x_a_param->set_name("permute_x_a_" + ts);
      permute_x_a_param->mutable_permute_param()->add_order(0);
      permute_x_a_param->mutable_permute_param()->add_order(1);
      permute_x_a_param->mutable_permute_param()->add_order(3);
      permute_x_a_param->mutable_permute_param()->add_order(2);
      permute_x_a_param->add_bottom("x_" + ts);
      permute_x_a_param->add_top("x_p_a_" + ts);
    }  // here is to change!
    {
      LayerParameter* att_x_param = net_param->add_layer();
      att_x_param->CopyFrom(biased_attention_param);
      att_x_param->set_name("att_x_" + tm1s);
      att_x_param->mutable_inner_product_param()->set_axis(3);
      att_x_param->add_bottom("x_p_a_" + ts);
      att_x_param->add_top("m_x_" + tm1s);
    }    //  fc layer ,change output,dim 
   {
      LayerParameter* permute_x_a_p_param = net_param->add_layer();
      permute_x_a_p_param->CopyFrom(permute_param);
      permute_x_a_p_param->set_name("permute_x_a_p_" + ts);
      permute_x_a_p_param->mutable_permute_param()->add_order(2);
      permute_x_a_p_param->mutable_permute_param()->add_order(0);
      permute_x_a_p_param->mutable_permute_param()->add_order(1);
      permute_x_a_p_param->mutable_permute_param()->add_order(3);
      permute_x_a_p_param->add_bottom("m_x_" + tm1s);
      permute_x_a_p_param->add_top("m_x_a_" + tm1s);
    }
    {
      LayerParameter* m_sum_layer = net_param->add_layer();
      m_sum_layer->CopyFrom(bias_layer_param);
      m_sum_layer->set_name("mask_input_" + ts);
      m_sum_layer->add_bottom("m_x_a_" + tm1s);
      m_sum_layer->add_bottom("m_" + tm1s);
      m_sum_layer->add_top("m_input_" + tm1s);
    }
   {
      LayerParameter* att_x_ap_param = net_param->add_layer();
      att_x_ap_param->CopyFrom(biased_attention_param);
      att_x_ap_param->set_name("att_x_ap_" + tm1s);
      att_x_ap_param->mutable_inner_product_param()->set_axis(3);
      att_x_ap_param->mutable_inner_product_param()->set_num_output(1);
      att_x_ap_param->add_bottom("m_input_" + tm1s);
      att_x_ap_param->add_top("m_x_ap_" + tm1s);  //256---->1
    }
    {
      LayerParameter* permute_m_param = net_param->add_layer();
      permute_m_param->CopyFrom(permute_param);
      permute_m_param->set_name("permute_m_" + ts);
      permute_m_param->mutable_permute_param()->add_order(1);
      permute_m_param->mutable_permute_param()->add_order(2);
      permute_m_param->mutable_permute_param()->add_order(0);
      permute_m_param->mutable_permute_param()->add_order(3);
      permute_m_param->add_bottom("m_x_ap_" + tm1s);
      permute_m_param->add_top("m_f_" + tm1s);  //10*8*30*1
    }
    // Add a softmax layers to generate attention masks
    {
      LayerParameter* softmax_m_param = net_param->add_layer();
      softmax_m_param->CopyFrom(softmax_param);
      softmax_m_param->mutable_softmax_param()->set_axis(2);
      softmax_m_param->set_name("softmax_m_" + tm1s);
      softmax_m_param->add_bottom("m_f_" + tm1s);
      softmax_m_param->add_top("mask_" + tm1s);
    }
   
     {
      LayerParameter* reshape_m_param = net_param->add_layer();
      reshape_m_param->CopyFrom(reshape_layer_param);
      BlobShape* shape = reshape_m_param->mutable_reshape_param()->mutable_shape();
      shape->Clear();
      shape->add_dim(0);
      shape->add_dim(0);
      shape->add_dim(0);
      reshape_m_param->set_name("reshape_m_" + tm1s);
      reshape_m_param->add_bottom("mask_" + tm1s);
      reshape_m_param->add_top("mask_reshape_" + tm1s);
    }
    //Reshape mask from 1*6*36 to 1*6*6*6
    /*
    {
      LayerParameter* reshape_param = net_param->add_layer();
      reshape_param->set_type("Reshape");
      BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
      new_shape->add_dim(1);  // One timestep.
      new_shape->add_dim(6);
      new_shape->add_dim(6);
      new_shape->add_dim(6);
      reshape_param->add_bottom("mask_" +tm1s);
      reshape_param->add_top("mask_reshape_" +tm1s);
    }*/
    // Conbine mask with input features
    {
      LayerParameter* scale_x_param = net_param->add_layer();
      scale_x_param->CopyFrom(scale_param);
      scale_x_param->set_name("scale_x_" + tm1s);
      scale_x_param->add_bottom("x_p_" + ts);
      scale_x_param->add_bottom("mask_reshape_" + tm1s);
      scale_x_param->add_top("x_mask_" + ts);
    }
 
    {
      LayerParameter* permute_x_mask_param = net_param->add_layer();
      permute_x_mask_param->CopyFrom(permute_param);
      permute_x_mask_param->set_name("permute_x_mask_" + ts);
      permute_x_mask_param->mutable_permute_param()->add_order(1);
      permute_x_mask_param->mutable_permute_param()->add_order(2);
      permute_x_mask_param->mutable_permute_param()->add_order(0);
      permute_x_mask_param->mutable_permute_param()->add_order(3);
      permute_x_mask_param->add_bottom("x_mask_" + ts);
      permute_x_mask_param->add_top("x_mask_p_" + ts);
    }
 
    {
      LayerParameter* reshape_x_param = net_param->add_layer();
      reshape_x_param->CopyFrom(reshape_param);
      reshape_x_param->set_name("reshape_x_" +ts);
      BlobShape* new_shape =
         reshape_x_param->mutable_reshape_param()->mutable_shape();
      new_shape->add_dim(this->N_);
      new_shape->add_dim(512);//512//384
      new_shape->add_dim(7);//7//6
      new_shape->add_dim(7);//7//6
      reshape_x_param->add_bottom("x_mask_p_" + ts);
      reshape_x_param->add_top("x_mask_reshape_"+ts);
    }
 
    {
      LayerParameter* pool_x_param = net_param->add_layer();
      pool_x_param->CopyFrom(pool_param);
      pool_x_param->set_name("pool_x_"+ts);
      pool_x_param->mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_SUM);
      pool_x_param->mutable_pooling_param()->set_kernel_size(7);//7//6
      pool_x_param->add_bottom("x_mask_reshape_"+ts);
      pool_x_param->add_top("x_pool_"+ts);
    }
 
    {
      LayerParameter* x_transform_param = net_param->add_layer();
      x_transform_param->CopyFrom(biased_hidden_param);
      x_transform_param->set_name("x_transform_" + ts);
      x_transform_param->add_param()->set_name("W_xc_" + ts);
      x_transform_param->add_param()->set_name("b_c" + ts);
      x_transform_param->add_bottom("x_pool_" +ts );
      x_transform_param->add_top("W_xc_x_"+ts);
    }
 
    {
      LayerParameter* x_transform_reshape_param = net_param->add_layer();
      x_transform_reshape_param->CopyFrom(reshape_param);
      x_transform_reshape_param->set_name("x_transform_reshape_" +ts);
      BlobShape* new_shape_r =
         x_transform_reshape_param->mutable_reshape_param()->mutable_shape();
      new_shape_r->add_dim(1);
      new_shape_r->add_dim(this->N_);
      new_shape_r->add_dim(num_output * 4);
      x_transform_reshape_param->add_bottom("W_xc_x_" + ts);
      x_transform_reshape_param->add_top("W_xc_x_r_"+ts);
    }
    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(sum_param);
      cont_h_param->mutable_eltwise_param()->set_coeff_blob(true);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }
 
    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }
 
    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_r_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static");
      }
      input_sum_layer->add_top("gate_input_" + ts);
    }
 
    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("cont_" + ts);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
    output_m_layer.add_bottom("mask_" + tm1s);
  }  // for (int t = 1; t <= this->T_; ++t)
 
  {
    LayerParameter* c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + this->int_to_str(this->T_));
    c_T_copy_param->add_top("c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
  net_param->add_layer()->CopyFrom(output_m_layer);
}
 
INSTANTIATE_CLASS(ALSTMLayer);
REGISTER_LAYER_CLASS(ALSTM);
 
}  // namespace caffe