package hw

import "time"

const SchemaVersion = 1

const (
	CaptureScopeInferenceHost = "inference_host"
	CaptureMethodDetected     = "detected"
)

type HardwareSnapshot struct {
	SchemaVersion   int                  `json:"schema_version"`
	CapturedAt      time.Time            `json:"captured_at"`
	Capture         HardwareCapture      `json:"capture"`
	Architecture    Architecture         `json:"architecture"`
	OperatingSystem OperatingSystem      `json:"operating_system"`
	Environment     ExecutionEnvironment `json:"environment"`
	CPU             CPU                  `json:"cpu"`
	Memory          SystemMemory         `json:"memory"`
	Accelerators    []Accelerator        `json:"accelerators"`
}

type HardwareCapture struct {
	Scope    string        `json:"scope"`
	Method   string        `json:"method"`
	Detector *DetectorInfo `json:"detector"`
}

type DetectorInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type Architecture struct {
	Name    string  `json:"name"`
	RawName *string `json:"raw_name"`
}

type OperatingSystem struct {
	Family    string  `json:"family"`
	Name      *string `json:"name"`
	Version   *string `json:"version"`
	Kernel    *string `json:"kernel"`
	RawFamily *string `json:"raw_family,omitempty"`
}

type ExecutionEnvironment struct {
	Kind    string  `json:"kind"`
	Name    *string `json:"name"`
	Version *string `json:"version"`
	RawKind *string `json:"raw_kind,omitempty"`
}

type CPU struct {
	Vendor             *string `json:"vendor"`
	Model              *string `json:"model"`
	SocketCount        *int    `json:"socket_count"`
	PhysicalCoreCount  *int    `json:"physical_core_count"`
	LogicalThreadCount *int    `json:"logical_thread_count"`
}

type SystemMemory struct {
	CapacityBytes uint64 `json:"capacity_bytes"`
}

type Accelerator struct {
	Index int `json:"index"`
	// DeviceIndex is the device's own index in its native enumeration (the
	// nvidia-smi GPU index for NVIDIA). It is the value a caller would use to
	// select this device (e.g. via CUDA_VISIBLE_DEVICES). It is nil when the
	// detector did not report one. This is independent of Index, which is a
	// display position in the sorted accelerator list.
	DeviceIndex     *int              `json:"device_index,omitempty"`
	Kind            string            `json:"kind"`
	RawKind         *string           `json:"raw_kind,omitempty"`
	Vendor          *string           `json:"vendor"`
	Model           *string           `json:"model"`
	Architecture    *string           `json:"architecture"`
	Memory          AcceleratorMemory `json:"memory"`
	Driver          *Driver           `json:"driver"`
	PowerLimitWatts *float64          `json:"power_limit_watts"`
}

type AcceleratorMemory struct {
	Kind          string  `json:"kind"`
	CapacityBytes *uint64 `json:"capacity_bytes"`
}

type Driver struct {
	Name    *string `json:"name"`
	Version *string `json:"version"`
}

type detectedAccelerator struct {
	identity string
	value    Accelerator
}
