# Mission UI 1.0

Visual ve TERCOM aynı `mission_ui` paketini kullanır: renkler, tipografi,
başlık, yöntem rozeti, durum göstergesi, metrik kartları ve alt kısayol çubuğu.
Üst bölüm görev kontrolleri; sol bölüm yöntem kanıtı; orta bölüm operasyon
haritası; sağ bölüm navigasyon ve telemetri için ayrılmıştır.

## Ortak kaynağı güncelleme

Paket iki repoda birebir aynı kaynakla bulunur ve her uygulamayla paketlenir.
Böylece uygulamalar ayrı bilgisayarlarda ve ayrı kurulumlarda çalışabilir;
kardeş klasöre, mutlak bir yola veya internet bağlantısına ihtiyaç duymaz.
İki repo aynı Python ortamına kurulacaksa `mission_ui` sürümleri aynı tutulmalıdır.

Şablonu değiştirdikten sonra, değiştirdiğiniz reponun kökünde çalıştırın:

```powershell
python -m mission_ui.sync 'C:\d_surucusu\tercom simulasyon'
python -m mission_ui.sync 'C:\d_surucusu\tercom simulasyon' --check
```

TERCOM tarafından güncelliyorsanız hedef olarak `C:\d_surucusu\simulasyon`
verin. `--check` dosya yazmaz ve fark olduğunda hata koduyla çıkar.
Yönteme özgü kontrolleri `mission_ui` içine taşımayın; algoritma, veri akışı,
harita koordinat dönüşümleri ve işçi yaşam döngüsü uygulamaların sorumluluğudur.

## Görünüm ve doğrulama

- Ortak zemin: `#0B1120`; paneller: `#111C2E`; vurgu: `#4DD9C0`.
- Başlık ve büyük değerler önceliklidir; ikincil açıklamalar daha sakin renktedir.
- Durumlar metinle ve renkle birlikte gösterilir; güven henüz ölçülmediyse
  başarılı konumlama izlenimi verilmez.
- Pencereler 1480 × 920 açılır, 1180 × 760 boyutuna kadar desteklenir.
- TERCOM: profil, harita, sürekli görünür metrikler; telemetri, benchmark ve
  kapsam sekmeleri. Uzun içerikler kaydırılabilir.
- Visual: gözlem, model ve eşleşme görüntüleri; harita; metrikler ve güven çubuğu.

Her iki repoda `python -m pytest tests -q` çalıştırın. Qt pencerelerini hem
varsayılan hem küçük boyutta kontrol edin. TERCOM'un `--fast` verisi sentetiktir;
görsel uygulamanın kaynak bekleme ekranı tam model/raster çalıştırma testi değildir.
Bu revizyonun ekran görüntüleri yerel `artifacts/ui-preview/` klasöründedir.
