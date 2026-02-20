import { useState, useEffect } from 'preact/hooks';
import { fetchJson, putJson } from '../api.js';
import { Section } from '../pages/intelligence/utils.jsx';
import TerminalToggle from './TerminalToggle.jsx';

const MODULE_SOURCES = {
  presence: [
    { key: 'camera_person', label: 'Camera Person Detection', description: 'Person detection via Frigate NVR' },
    { key: 'camera_face', label: 'Camera Face Recognition', description: 'Face recognition via Frigate NVR' },
    { key: 'motion', label: 'Motion Sensors', description: 'Binary sensor motion detectors' },
    { key: 'light_interaction', label: 'Light Interaction', description: 'Light state changes indicating presence' },
    { key: 'dimmer_press', label: 'Dimmer Press', description: 'Physical dimmer switch presses' },
    { key: 'door', label: 'Door Sensors', description: 'Door open/close events' },
    { key: 'media_active', label: 'Media Active', description: 'Media player playing/paused state' },
    { key: 'media_inactive', label: 'Media Inactive', description: 'Media player off/standby state' },
    { key: 'device_tracker', label: 'Device Tracker', description: 'WiFi/Bluetooth device tracking' },
  ],
  activity: [
    { key: 'light', label: 'Lights', description: 'Light entity state changes' },
    { key: 'switch', label: 'Switches', description: 'Switch entity state changes' },
    { key: 'binary_sensor', label: 'Binary Sensors', description: 'Binary sensor state changes' },
    { key: 'media_player', label: 'Media Players', description: 'Media player state changes' },
    { key: 'climate', label: 'Climate', description: 'Thermostat and climate controls' },
    { key: 'cover', label: 'Covers', description: 'Window blinds and garage doors' },
  ],
  anomaly: [
    { key: 'autoencoder', label: 'Autoencoder', description: 'Neural network reconstruction error detection' },
    { key: 'isolation_forest', label: 'Isolation Forest', description: 'Tree-based outlier detection' },
  ],
  shadow: [
    { key: 'can_predict', label: 'Predictable Capabilities', description: 'Capabilities with can_predict=true' },
  ],
  discovery: [
    { key: 'light', label: 'Lights', description: 'Discover light entities' },
    { key: 'switch', label: 'Switches', description: 'Discover switch entities' },
    { key: 'binary_sensor', label: 'Binary Sensors', description: 'Discover binary sensor entities' },
    { key: 'media_player', label: 'Media Players', description: 'Discover media player entities' },
    { key: 'climate', label: 'Climate', description: 'Discover climate entities' },
    { key: 'cover', label: 'Covers', description: 'Discover cover entities' },
  ],
};

/**
 * Data source toggle list for a given module.
 * Fetches enabled sources from the API and renders a toggle per source definition.
 * Changes are written back via PUT immediately on toggle.
 *
 * @param {Object} props
 * @param {string} props.module - Module key (e.g. 'presence', 'activity')
 * @param {string} props.title - Section title
 * @param {string} [props.subtitle] - Section subtitle
 */
export default function DataSourceConfig({ module, title, subtitle }) {
  const [enabledSources, setEnabledSources] = useState(null);
  const [saving, setSaving] = useState(false);
  const [warning, setWarning] = useState(null);

  const sourceDefs = MODULE_SOURCES[module] || [];

  useEffect(() => {
    fetchJson(`/api/config/modules/${module}/sources`)
      .then((data) => setEnabledSources(data.sources || []))
      .catch((err) => {
        console.warn(`DataSourceConfig: failed to load sources for ${module}`, err);
        // Default to all sources enabled if API unavailable
        setEnabledSources(sourceDefs.map((src) => src.key));
      });
  }, [module]);

  async function handleToggle(sourceKey, newEnabled) {
    if (!enabledSources) return;

    const currentEnabled = enabledSources.includes(sourceKey);

    // Safety: prevent disabling the last enabled source
    if (currentEnabled && !newEnabled && enabledSources.length <= 1) {
      setWarning('At least one source must remain enabled.');
      setTimeout(() => setWarning(null), 3000);
      return;
    }

    const updated = newEnabled
      ? [...enabledSources, sourceKey]
      : enabledSources.filter((src) => src !== sourceKey);

    setEnabledSources(updated);

    setSaving(true);
    try {
      await putJson(`/api/config/modules/${module}/sources`, { sources: updated });
    } catch (err) {
      console.error(`DataSourceConfig: failed to save sources for ${module}`, err);
      // Roll back on failure
      setEnabledSources(enabledSources);
    } finally {
      setSaving(false);
    }
  }

  if (sourceDefs.length === 0) return null;

  const enabledCount = enabledSources ? enabledSources.filter((src) => sourceDefs.some((def) => def.key === src)).length : 0;
  const summary = enabledSources ? `${enabledCount}/${sourceDefs.length} active` : null;

  return (
    <Section
      title={title}
      subtitle={subtitle}
      summary={summary}
      defaultOpen={false}
      loading={enabledSources === null}
    >
      {warning && (
        <div class="text-xs mb-3 px-3 py-2" style="background: var(--status-warning-glow); color: var(--status-warning); border-radius: var(--radius)">
          {warning}
        </div>
      )}
      <div class="space-y-1">
        {sourceDefs.map((src) => {
          const isEnabled = enabledSources ? enabledSources.includes(src.key) : true;
          return (
            <div
              key={src.key}
              class="flex flex-col sm:flex-row sm:items-center gap-2 py-3"
              style="border-bottom: 1px solid var(--border-subtle)"
            >
              <div class="flex-1 min-w-0">
                <span class="text-sm font-medium" style="color: var(--text-secondary)">{src.label}</span>
                {src.description && (
                  <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">{src.description}</p>
                )}
              </div>
              <div class="flex items-center gap-2 flex-shrink-0">
                {saving && <span class="text-xs" style="color: var(--accent)">Saving...</span>}
                <TerminalToggle
                  enabled={isEnabled}
                  onChange={(val) => handleToggle(src.key, val)}
                />
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}
