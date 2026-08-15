import numpy as np
from typing import Optional, Tuple, Any

class SlidingSpatialDomainWindow:
    r"""
    Sliding Spatial Domain Window z Automatyczną Normalizacją Afiniczną dla Wielomianów Czebyszewa.
    
    Mapowanie afiniczne: X \in [X_min, X_max] -> \hat{X} \in [-1, 1].
    Skalowanie analitycznych gradientów według reguły łańcuchowej:
    \frac{\partial f}{\partial X_d} = s_d \cdot \frac{\partial f}{\partial \hat{X}_d},  gdzie s_d = \frac{2}{X_{max, d} - X_{min, d}}.
    """
    def __init__(
        self,
        spatial_dim: int,
        initial_min: Optional[np.ndarray] = None,
        initial_max: Optional[np.ndarray] = None,
        margin: float = 0.05
    ):
        self.spatial_dim = spatial_dim
        self.margin = margin
        
        if initial_min is None:
            self.domain_min = -1.0 * np.ones(spatial_dim)
        else:
            self.domain_min = np.array(initial_min, dtype=np.float64)
            
        if initial_max is None:
            self.domain_max = 1.0 * np.ones(spatial_dim)
        else:
            self.domain_max = np.array(initial_max, dtype=np.float64)
            
        self._check_and_fix_bounds()

    def _check_and_fix_bounds(self):
        diff = self.domain_max - self.domain_min
        small_mask = diff < 1e-6
        if np.any(small_mask):
            self.domain_max[small_mask] = self.domain_min[small_mask] + 1.0

    def update_bounds(self, X: np.ndarray, mode: str = "expand", ema_alpha: float = 0.1):
        """
        Aktualizacja ruchomego okna przestrzennego na podstawie obserwowanych punktów X.
        mode: 'expand' (rozszerzanie granic) lub 'ema' (wygładzanie wyznaczonego okna).
        """
        assert X.shape[1] == self.spatial_dim
        obs_min = np.min(X, axis=0) - self.margin
        obs_max = np.max(X, axis=0) + self.margin
        
        if mode == "expand":
            self.domain_min = np.minimum(self.domain_min, obs_min)
            self.domain_max = np.maximum(self.domain_max, obs_max)
        elif mode == "ema":
            self.domain_min = (1.0 - ema_alpha) * self.domain_min + ema_alpha * obs_min
            self.domain_max = (1.0 - ema_alpha) * self.domain_max + ema_alpha * obs_max
        elif mode == "fit":
            self.domain_min = obs_min
            self.domain_max = obs_max
            
        self._check_and_fix_bounds()

    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""
        Przekształcenie afiniczne z dowolnej przestrzeni X do \hat{X} \in [-1, 1].
        """
        diff = self.domain_max - self.domain_min
        X_hat = 2.0 * (X - self.domain_min) / diff - 1.0
        return np.clip(X_hat, -1.0, 1.0)

    def inverse_transform(self, X_hat: np.ndarray) -> np.ndarray:
        r"""
        Przekształcenie odwrotne z \hat{X} \in [-1, 1] do oryginalnej przestrzeni X.
        """
        diff = self.domain_max - self.domain_min
        return self.domain_min + 0.5 * (X_hat + 1.0) * diff

    def get_scale_factors(self) -> np.ndarray:
        r"""
        Zwraca wektor czynników skalujących s_d = d\hat{X}_d / dX_d = 2 / (X_max,d - X_min,d).
        """
        return 2.0 / (self.domain_max - self.domain_min)


class NormalizedKANField:
    r"""
    Wrapper Pola KAN integrujący SlidingSpatialDomainWindow.
    Zapewnia całkowite bezpieczeństwo numeryczne dla dowolnych dziedzin przestrzennych oraz
    analityczną precyzję gradientów dla modeli KAN.
    """
    def __init__(self, base_model: Any, domain_window: Optional[SlidingSpatialDomainWindow] = None):
        self.base_model = base_model
        self.spatial_dim = getattr(base_model, "spatial_dim", 10)
        
        if domain_window is not None:
            self.domain_window = domain_window
        else:
            self.domain_window = SlidingSpatialDomainWindow(self.spatial_dim)

    def evaluate(self, X: np.ndarray, auto_update_domain: bool = False) -> np.ndarray:
        """
        Ewaluacja pola z automatyczną normalizacją współrzędnych.
        """
        if auto_update_domain:
            self.domain_window.update_bounds(X, mode="expand")
            
        X_hat = self.domain_window.transform(X)
        return self.base_model.evaluate(X_hat)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        r"""
        Analityczny gradient \nabla_X f(X) skalowany według reguły łańcuchowej s_d.
        """
        X_hat = self.domain_window.transform(X)
        grad_hat = self.base_model.gradient(X_hat) # (N, D)
        scale_factors = self.domain_window.get_scale_factors() # (D,)
        
        # \nabla_X f = \nabla_{\hat{X}} f \odot s
        return grad_hat * scale_factors
