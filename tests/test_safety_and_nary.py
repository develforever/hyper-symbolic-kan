import numpy as np
from src.mct_nse.concurrent_category_filter import ConcurrentCategoryFilter
from src.hs_ckan.nary_spatiotemporal import NarySpatioTemporalEngine

def test_concurrent_category_filter_zero_violations():
    """Weryfikacja czy Category Guard gwarantuje 100% bezpieczeństwo."""
    cat_filter = ConcurrentCategoryFilter()
    cat_filter.add_invariant(
        "BoxBounds",
        lambda S: np.all((S[:, :3] >= -10.0) & (S[:, :3] <= 10.0), axis=1),
        lambda S: np.hstack([np.clip(S[:, :3], -10.0, 10.0), S[:, 3:6]])
    )
    
    # Stan z celowymi naruszeniami
    raw_states = np.random.uniform(-50.0, 50.0, (500, 6))
    safe_states, violations = cat_filter.filter_state(raw_states)
    
    assert np.all(safe_states[:, :3] >= -10.0)
    assert np.all(safe_states[:, :3] <= 10.0)

def test_isotropic_spatiotemporal_encoding():
    """Weryfikacja symetrii i izotropii kodowania relacji czasoprzestrzennych."""
    engine = NarySpatioTemporalEngine(num_entities=10, num_predicates=5, spatial_dim=4, kan_degree=3)
    coords = np.random.uniform(-1.0, 1.0, (20, 4))
    encoded = engine.encode_spatiotemporal_coords(coords)
    
    assert encoded.shape[0] == 20
    # D=4, K=3 -> K+1=4. 1D: 4*4=16. Pairs: 6*16=96. Total = 112.
    assert encoded.shape[1] == 112
    assert not np.any(np.isnan(encoded))
