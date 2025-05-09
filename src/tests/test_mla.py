import unittest
import torch
from ..mla import MultiHeadLatentAttention  # Using relative import

class TestMultiLatentAttention(unittest.TestCase):
    def setUp(self):
        # Common dimensions for testing
        self.d_model = 512
        self.num_head = 8
        self.d_embed = 512
        self.d_c = 64  # Compression dim for K/V
        self.d_c1 = 64  # Compression dim for Q
        self.d_rotate = 32  # For future RoPE implementation
        self.batch_size = 2
        self.seq_len = 10
        
        # Initialize MLA
        self.mla = MultiHeadLatentAttention(
            d_model=self.d_model,
            num_head=self.num_head,
            d_embed=self.d_embed,
            d_c=self.d_c,
            d_c1=self.d_c1,
            d_rotate=self.d_rotate
        )
        
    def test_basic_forward(self):
        """Test basic forward pass without caching"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.mla(x)
        
        # Check output shape
        self.assertEqual(
            output.shape, 
            (self.batch_size, self.seq_len, self.d_model),
            "Output shape mismatch"
        )
        
    def test_cross_attention(self):
        """Test cross-attention functionality"""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        kv = torch.randn(self.batch_size, self.seq_len * 2, self.d_model)  # Different seq_len
        
        output = self.mla(query, key_value_states=kv)
        self.assertEqual(
            output.shape, 
            (self.batch_size, self.seq_len, self.d_model),
            "Cross-attention output shape mismatch"
        )
        
    def test_cache_initialization(self):
        """Test if cache is properly initialized"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _ = self.mla(x, use_cache=True, start_pos=0)
        
        self.assertIsNotNone(self.mla.cache_kv)
        self.assertEqual(
            self.mla.cache_kv.shape[-1],
            self.d_c,
            "Cache compression dimension mismatch"
        )
        
    def test_sequential_caching(self):
        """Test sequential forward passes with caching"""
        # Initial sequence
        prompt_len = 5
        prompt = torch.randn(self.batch_size, prompt_len, self.d_model)
        
        # First forward pass with prompt
        output1 = self.mla(prompt, use_cache=True, start_pos=0)
        cached_kv_1 = self.mla.cache_kv[:, :prompt_len].clone()
        
        # Second forward pass with one new token
        new_token = torch.randn(self.batch_size, 1, self.d_model)
        output2 = self.mla(new_token, use_cache=True, start_pos=prompt_len)
        
        # Verify cache consistency
        # First part of cache should remain unchanged
        self.assertTrue(
            torch.allclose(
                self.mla.cache_kv[:, :prompt_len],
                cached_kv_1,
                rtol=1e-5
            ),
            "Cache was modified for previously processed tokens"
        )
        
        # Verify new token was added to cache
        self.assertFalse(
            torch.allclose(
                self.mla.cache_kv[:, prompt_len:prompt_len+1],
                torch.zeros_like(self.mla.cache_kv[:, prompt_len:prompt_len+1]),
                rtol=1e-5
            ),
            "New token was not added to cache"
        )
        
    def test_attention_mask_with_cache(self):
        """Test attention masking with cached KV"""
        seq_len = 5
        x = torch.randn(self.batch_size, seq_len, self.d_model)
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'), 
            diagonal=1
        ).unsqueeze(0)
        
        # First forward pass with mask
        output1 = self.mla(x, use_cache=True, start_pos=0, att_mask=mask)
        
        # Second pass with one token
        new_token = torch.randn(self.batch_size, 1, self.d_model)
        extended_mask = torch.triu(
            torch.ones(seq_len + 1, seq_len + 1) * float('-inf'),
            diagonal=1
        ).unsqueeze(0)
        
        output2 = self.mla(
            new_token,
            use_cache=True,
            start_pos=seq_len,
            att_mask=extended_mask
        )
        
        self.assertEqual(
            output2.shape,
            (self.batch_size, 1, self.d_model),
            "Output shape incorrect for cached attention with mask"
        )

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiLatentAttention)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

# Run the tests
run_tests()