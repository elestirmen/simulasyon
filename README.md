# GPS’siz yerelleştirme simülasyonu

Bu depo, İHA’nın gözlem haritasından alınan **üç komşu kırpıntı** ile referans haritada **şablon eşleştirme** yaparak konum tahmini sürecini simüle eder. Ana uygulama OpenCV penceresinde gözlem alanı, model çıktıları ve referans harita önizlemesini bir arada gösteren bir **dashboard** sunar.

## Gereksinimler

- Python 3.x
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [Rasterio](https://rasterio.readthedocs.io/) (GeoTIFF okuma / hizalama)
- [pyproj](https://pyproj4.github.io/pyproj/)
- [TensorFlow / Keras](https://www.tensorflow.org/) (`.h5` model yükleme)

Örnek kurulum:

```bash
pip install opencv-python numpy rasterio pyproj tensorflow
```

> GPU kullanımı isteğe bağlıdır; TensorFlow CPU sürümü de çalışır, ancak model çıkarımı daha yavaş olabilir.

## Ana program: üçlü dashboard

```bash
python simulasyon_yonlendirme_uclu_dashboard.py
```

Çalıştırmadan önce `simulasyon_yonlendirme_uclu_dashboard.py` içindeki `SimulationConfig` sınıfında tanımlı **dosya yollarının** makinenizdeki harita, parça ve model dosyalarına işaret etmesi gerekir (varsayılanlar Ürgüp örneğine göre yazılmıştır):

| Alan | Açıklama |
|------|----------|
| `reference_map_path` | Referans harita (GeoTIFF veya gri ton görüntü) |
| `observation_map_path` | Gözlem / canlı kaynak haritası |
| `observation_georef_path` | Gözlem jeodezi (çoğu kurulumda gözlem haritası ile aynı) |
| `observation_grid_georef_path` | İsteğe bağlı; referans ızgarasına hizalama için |
| `dem_path` | **Sadece `scenario_mode == "irtifa"`** iken irtifa simülasyonu için DEM |
| `model_path` | Keras `.h5` segmentasyon / çıkarım modeli |

`scenario_mode`:

- `"normal"` — Sabit irtifa varsayımı (DEM yüklenmez).
- `"irtifa"` — AGL/ölçek ve arazi yüksekliği için DEM kullanılır.

### Klavye ve arayüz

- **W A S D**: gözlem imlecini hareket
- **Q / E**: sola / sağa dönüş
- **+ / −** (irtifa senaryosunda): irtifa adımı
- **ESC** veya **X**: çıkış
- **H**: panel daraltma; **B T O R Y G**: bilgi, trajektori, ROI çerçevesi, TM kutuları, yön oku, gözlem kutuları (dashboard üzerindeki kısayollar ile uyumlu)

İsteğe bağlı **otonom mod** ve **görev benchmark**ları `SimulationConfig` içindeki `autonomous_mode_enabled`, `mission_benchmark_enabled`, `diagnostic_benchmark_enabled` bayrakları ile açılır; çıktılar `diagnostic_output_dir` altına yazılabilir.

## Yardımcı modül

- **`gps_denied_autonomy.py`**: Görev senaryoları, yerelleştirme kalitesi ve otonom adım seçimi için veri yapıları ve yardımcı fonksiyonlar (dashboard tarafından import edilir).

## Diğer betikler (daha eski / özel denemeler)

Aynı klasörde tek başına çalışan veya farklı deneyler için kullanılan örnekler:

- `simulasyon_yonlendirme.py`, `simulasyon_yonlendirme_uclu.py` — yönlendirme / şablon eşleştirme denemeleri
- `simulasyon_yonlendirme_model_okuma.py` — model okuma odaklı akış
- `simulasyon_otonom.py`, `simulasyon_konuma_otonom_gitme.py`, `simulasyon_konuma_otonom_gitme_HIZLI.py`, `simulasyon_hizli.py` — otonom / hızlı varyantlar
- `template_matching_dongu.py`, `image_rotate.py`, `image_rotate_funcs.py` — şablon eşleştirme ve görüntü dönüş yardımcıları

## Veri ve `.gitignore`

Büyük raster ve model dosyaları genelde repoda tutulmaz; `.gitignore` yalnızca belirli uzantıları izin verir. Harita ve model dosyalarını yerel olarak `haritalar/`, `parcalar/` gibi dizinlere koyup yolları `SimulationConfig` ile eşleştirmeniz gerekir.

## Ek not

GPS’siz otonomi ve kalite mantığına dair inceleme notları için depodaki `GPS_DENIED_REVIEW.md` dosyasına bakabilirsiniz.
