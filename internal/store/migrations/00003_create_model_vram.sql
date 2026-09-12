-- +goose Up
-- Measured VRAM requirement per model, recorded on a successful load and used
-- as the fallback when models.*.vramMB is not declared.
CREATE TABLE model_vram (
    model      TEXT    NOT NULL PRIMARY KEY,
    vram_mb    INTEGER NOT NULL,
    ts_updated TEXT    NOT NULL
);

-- +goose Down
DROP TABLE model_vram;
