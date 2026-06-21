-- ==========================================================
-- SKRIP MIGRASI DATABASE db_kutamasya (VERSI ULTRA AMAN)
-- ==========================================================

-- 1. Membuat tabel baru 'package_vectors' untuk menyimpan representasi fitur teks
--    dan vektor TF-IDF dari masing-masing paket wisata yang aktif.
CREATE TABLE IF NOT EXISTS `package_vectors` (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `package_id` bigint UNSIGNED NOT NULL,
  `combined_features` text DEFAULT NULL,                  -- Fitur teks gabungan setelah preprocessing
  `tfidf_vector` json DEFAULT NULL,                       -- Array angka vektor TF-IDF
  `vocabulary_hash` varchar(64) DEFAULT NULL,             -- Hash untuk verifikasi konsistensi vocabulary
  `last_updated` timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `package_vectors_package_id_unique` (`package_id`),
  CONSTRAINT `fk_package_vectors_tour_packages` 
    FOREIGN KEY (`package_id`) REFERENCES `tour_packages`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Memodifikasi tabel 'recommendations' yang sudah dikelola oleh Laravel
--    menggunakan Stored Procedure agar aman dari galat duplikasi (IF NOT EXISTS).
DROP PROCEDURE IF EXISTS migrate_recommendations;

DELIMITER //
CREATE PROCEDURE migrate_recommendations()
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = 'db_kutamasya' 
      AND TABLE_NAME = 'recommendations' 
      AND COLUMN_NAME = 'results'
  ) THEN
    ALTER TABLE `recommendations` 
      ADD COLUMN `results` json DEFAULT NULL 
        COMMENT 'Daftar ID paket wisata hasil rekomendasi (format JSON array)';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = 'db_kutamasya' 
      AND TABLE_NAME = 'recommendations' 
      AND COLUMN_NAME = 'similarity_scores'
  ) THEN
    ALTER TABLE `recommendations` 
      ADD COLUMN `similarity_scores` json DEFAULT NULL 
        COMMENT 'Skor kemiripan cosine similarity masing-masing paket (format JSON)';
  END IF;
END //
DELIMITER ;

CALL migrate_recommendations();
DROP PROCEDURE IF EXISTS migrate_recommendations;
