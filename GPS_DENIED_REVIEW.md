## GPS-Denied Review

### Mevcut yaklaşım
Proje, gözlem haritasından üç pencere çıkarıp model ile üç template üretir ve bunları referans haritada `matchTemplate` ile arar. Eşleşen kutuların kesişimi konum tahmini olarak kullanılır.

### Temel zayıflıklar
- Güven skoru olmadan tek-frame karar verildiği için hatalı eşleşme sonrası ROI kolayca yanlış yere kilitlenebilir.
- Görev mantığı büyük ölçüde operatör komutuna bağlıydı; waypoint takibi ve yeniden bulma davranışı eksikti.
- Nominal test noktaları vardı, fakat görev seviyesinde rota başarısı, yeniden bulma kabiliyeti ve düşük güven durumları ölçülmüyordu.
- Önceki diagonal üçlü örnekleme, benzer dokularda ayrıştırıcılığı sınırlıyordu.

### Eklenen iyileştirmeler
- Yerelleştirme için güven skoru ve düşük güvene bağlı ROI büyütme eklendi.
- Takip merkezi, ölçüm güvenine göre yumuşatılarak güncelleniyor.
- Üçlü örnekleme diagonal yerine daha ayırt edici üçgen geometriye çekildi.
- Görev başlangıcı için bilinen launch konumundan bootstrap desteği eklendi.
- Otonom waypoint modu eklendi.
- Otonom benchmark senaryoları eklendi:
  - `baseline_box_route`
  - `heading_bias_zigzag`
  - `altitude_transition_route`
  - `reacquire_after_dropout`

### Mühendislik yorumu
- Yazılım mühendisi gözüyle: proje artık algı, takip ve görev mantığını daha temiz katmanlara ayırıyor.
- İHA mühendisi gözüyle: düşük güven durumunda agresif ilerlemek yerine yeniden kazanım davranışı kullanılıyor; başlangıçta bilinen kalkış bölgesi varsa görsel takip daha kararlı başlıyor.
- Uçak mühendisi gözüyle: heading ve irtifa değişiminin görev başarısına etkisi artık senaryo seviyesinde ölçülebiliyor.
- Bilimsel gözle: çıktı yalnızca görsel değil; rota bazlı başarı, hata ve güven metrikleri JSON olarak dışa aktarılıyor.

### Çalıştırma
- Dashboard manuel kullanım: `simulasyon_yonlendirme_uclu_dashboard.py`
- Otonom waypoint modu için:
  - `SimulationConfig.autonomous_mode_enabled = True`
- Bilinen başlangıç konumundan başlatmak için:
  - `SimulationConfig.bootstrap_tracking_from_start = True`
- Benchmark için:
  - `SimulationConfig.mission_benchmark_enabled = True`
  - Sadece benchmark çalıştırmak için `SimulationConfig.mission_benchmark_only = True`

Benchmark çıktıları `diagnostics/mission_bench_*` klasörüne yazılır.
