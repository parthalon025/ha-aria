import { useState, useEffect, useRef } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import { Section } from '../pages/intelligence/utils.jsx';

/**
 * Inline settings panel — shows tunable parameters for a specific category.
 * Embeds directly on OODA pages so users can adjust settings in context.
 *
 * @param {Object} props
 * @param {string[]} props.categories - Config categories to show (e.g. ['Activity Monitor'])
 * @param {string} [props.title] - Section title (default: 'Settings')
 * @param {string} [props.subtitle] - Section subtitle
 */
export default function InlineSettings({ categories, title, subtitle }) {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);

  async function fetchConfigs() {
    try {
      const data = await fetchJson('/api/config');
      const filtered = (data.configs || []).filter(
        (cfg) => categories.includes(cfg.category)
      );
      setConfigs(filtered);
    } catch (err) {
      // Silently fail — settings are optional, not critical
      console.warn('InlineSettings: failed to load config', err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchConfigs(); }, []);

  if (loading || configs.length === 0) return null;

  return (
    <Section
      title={title || 'Settings'}
      subtitle={subtitle || 'Adjust parameters that affect this page. Changes apply immediately.'}
      defaultOpen={false}
    >
      <div class="space-y-1">
        {configs.map((cfg) => (
          <ParamRow key={cfg.key} config={cfg} onUpdate={fetchConfigs} />
        ))}
      </div>
    </Section>
  );
}

function ParamRow({ config, onUpdate }) {
  const [value, setValue] = useState(config.value);
  const [saving, setSaving] = useState(false);
  const timerRef = useRef(null);
  const isDefault = config.value === config.default_value;

  useEffect(() => { setValue(config.value); }, [config.value]);

  function debouncedSave(newVal) {
    setValue(newVal);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        await putJson(`/api/config/${config.key}`, { value: String(newVal) });
        onUpdate();
      } catch (err) {
        console.error('Config save failed:', err);
      } finally {
        setSaving(false);
      }
    }, 500);
  }

  async function handleReset() {
    setSaving(true);
    try {
      await postJson(`/api/config/reset/${config.key}`, {});
      setValue(config.default_value);
      onUpdate();
    } catch (err) {
      console.error('Config reset failed:', err);
    } finally {
      setSaving(false);
    }
  }

  const vtype = config.value_type;

  return (
    <div class="flex flex-col sm:flex-row sm:items-center gap-2 py-3" style="border-bottom: 1px solid var(--border-subtle)">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium" style="color: var(--text-secondary)">{config.label || config.key}</span>
          {saving && <span class="text-xs" style="color: var(--accent)">Saving...</span>}
        </div>
        {config.description && (
          <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">{config.description}</p>
        )}
      </div>
      <div class="flex items-center gap-2 flex-shrink-0">
        {vtype === 'number' && (
          <div class="flex items-center gap-2">
            <input
              type="range"
              min={config.min_value ?? 0}
              max={config.max_value ?? 100}
              step={config.step ?? 1}
              value={value}
              onInput={(event) => debouncedSave(event.target.value)}
              class="w-28"
              style="accent-color: var(--accent)"
            />
            <span class="text-sm w-16 text-right data-mono" style="color: var(--text-secondary)">{value}</span>
          </div>
        )}
        {vtype === 'boolean' && (
          <button
            class="relative w-10 h-5 rounded-full transition-colors"
            style={value === 'true' || value === '1' ? 'background: var(--accent)' : 'background: var(--bg-inset)'}
            onClick={() => debouncedSave(value === 'true' || value === '1' ? 'false' : 'true')}
          >
            <span class={`absolute top-0.5 w-4 h-4 rounded-full shadow transition-transform ${
              value === 'true' || value === '1' ? 'translate-x-5' : 'translate-x-0.5'
            }`} style="background: var(--bg-surface)" />
          </button>
        )}
        {vtype === 'string' && (
          <input
            type="text"
            value={value}
            onInput={(event) => debouncedSave(event.target.value)}
            class="t-input px-2 py-1 text-sm w-48"
          />
        )}
        {vtype === 'select' && (
          <select
            value={value}
            onChange={(event) => debouncedSave(event.target.value)}
            class="t-input px-2 py-1 text-sm"
          >
            {(config.options || '').split(',').map((option) => (
              <option key={option.trim()} value={option.trim()}>{option.trim()}</option>
            ))}
          </select>
        )}
        {!isDefault && (
          <button
            onClick={handleReset}
            class="text-xs px-2 py-1"
            style="border-radius: var(--radius); color: var(--accent); cursor: pointer;"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
