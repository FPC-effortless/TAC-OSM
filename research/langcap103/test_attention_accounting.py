"""Constructed arithmetic and double-count falsifiers, no model fitting."""
import unittest
from attention_accounting import FORWARD, BACKWARD, dense_attention_units, supplement_record


class AttentionAccountingTests(unittest.TestCase):
    def test_four_layer_1m_forward_and_backward_matrix_units(self):
        shape=[2,8,256,16]
        f=dense_attention_units(shape,shape,shape)
        b=dense_attention_units(shape,shape,shape,backward=True)
        self.assertEqual(4*(f+b),805306368)

    def test_eight_layer_10m_forward_and_backward_matrix_units(self):
        shape=[2,8,256,32]
        self.assertEqual(8*(dense_attention_units(shape,shape,shape)+
                           dense_attention_units(shape,shape,shape,backward=True)),3221225472)

    def test_wrong_head_or_key_shape_rejected(self):
        with self.assertRaises(ValueError):
            dense_attention_units([2,8,256,16],[2,4,256,16],[2,8,256,16])
        with self.assertRaises(ValueError):
            dense_attention_units([2,8,256,16],[2,8,256,16],[2,8,255,16])

    def test_runtime_schema_positions_and_double_count_fail_closed(self):
        r={"operator":BACKWARD,"schema_argument_names":["grad_out","query","key","value"],
           "observed_input_shapes":[[2,8,256,16]]*4,"already_profiled_flops":0,"dense_matmul_children":[]}
        self.assertEqual(supplement_record(r),134217728)
        with self.assertRaises(ValueError):supplement_record({**r,"already_profiled_flops":1})
        with self.assertRaises(ValueError):supplement_record({**r,"dense_matmul_children":["aten::bmm"]})


if __name__=='__main__':unittest.main()
