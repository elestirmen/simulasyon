## GPS-Denied Review

### Mevcut yaklaşım
Proje, gözlem haritasından üç pencere çıkarıp model ile üç template üretir ve bunları referans haritada `matchTemplate` ile arar. Eşleşen kutuların kesişimi konum tahmini olarak kullanılır.

### Temel zayıflıklar
- Güven skoru olmadan tek-frame karar verildiğinde hatalı eşleşme sonrası ROI kolayca yanlış yere kilitlenebilir.
- Görev mantığı operatör komutuna bağlıydı; waypoint takibi ve düşük güvende yeniden kazanım davranışı eksikti.
- Tek-adım görsel sıçramalar (yanlış eşleşme) takip merkezini ani biçimde kaydırabiliyordu.

### Eklenen iyileştirmeler
Aşağıdakiler `gps_denied_autonomy.py` içinde tanımlı ve dashboard ana döngüsünde aktif kullanılır:
- **Lokalizasyon kalitesi** (`compute_localization_quality`): normalize skorlar, `score_floor` / `score_mean`, merkez yayılımı (`center_spread_px`), birleşik `confidence` ve `is_reliable` bayrağı (eşikler `localization_*_threshold`).
- **Düşük güvene bağlı ROI büyütme** (`update_search_window_size`): katı üçlü hizalama sağlandığında pencere tabana döner; aksi halde `search_window_growth_step` / `search_window_failure_growth` ile büyür.
- **Sensör füzyonu** (`fuse_measurement_with_prior`): takip merkezi ölçüm güvenine göre yumuşatılır; `max_visual_jump_px` eşiğini aşan sıçramalar reddedilir.
- **Kalman filtresi** (`PositionKalmanFilter`, K tuşu / `kalman_enabled`): sabit-hız modelli 2D konum filtresi; güvenilir ölçümlerde güncellenir.
- **Otonom waypoint modu** (`choose_autonomous_action`, `update_waypoint_progress`): P tuşu ile açılır, fare ile harita üzerinde hedef seçilir; gövde-ekseni hizalama, ardışık kabul ve takılma (stuck) kurtarma içerir.

> Not: Üçlü örnekleme hâlâ **diagonal** geometridedir (`get_observation_boxes`); offset vektörü başlık açısıyla döndürülür ama üç pencere eş-doğrusal kalır.

### Tanılama (diagnostic) toplu çalıştırma
Dashboard, üçlü şablon kalitesini ölçen bir tanılama modu içerir (`run_template_diagnostics`):
- `SimulationConfig.diagnostic_benchmark_enabled = True` → başlangıçta çalışır.
- `SimulationConfig.diagnostic_benchmark_only = True` → çıktı yazıldıktan sonra dashboard açılmadan çıkar.
- Tohum noktaları: `SimulationConfig.diagnostic_benchmark_points`.

Çıktılar `diagnostics/template_diag_YYYYMMDD_HHMMSS/` altına yazılır: her vaka için `case_XX_..._triptych.png`, `case_XX_..._meta.json` ve `summary.json`.

### Mühendislik yorumu
- Yazılım mühendisi gözüyle: algı (`localize_template_triplet`), kalite/füzyon (`gps_denied_autonomy`) ve görev mantığı (otonom döngü) ayrı katmanlara ayrılmış durumda.
- İHA mühendisi gözüyle: düşük güvende agresif ilerleme yerine dönüş/yeniden kazanım tercih ediliyor; Kalman açıkken arama çerçevesi filtre konumuna odaklanarak tek-adım hatalarına dayanıklılık artıyor.
- Bilimsel gözle: her adım CSV'ye (`log_simulasyon_*.csv`) skor, güven, yayılım, ham/Kalman hata (px ve m) olarak yazılır; tanılama vakaları PNG/JSON olarak dışa aktarılır.

### Çalıştırma
- Manuel dashboard: `python simulasyon_yonlendirme_uclu_dashboard.py`
- Otonom waypoint modu: `SimulationConfig.autonomous_mode_enabled = True` (veya çalışırken **P**).
- Tanılama: `SimulationConfig.diagnostic_benchmark_enabled = True` (yalnız tanılama için ayrıca `diagnostic_benchmark_only = True`).
