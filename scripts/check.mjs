import { execFileSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const run = (command, args, options = {}) => execFileSync(command, args, { cwd: root, stdio: 'inherit', ...options })

run(process.execPath, ['scripts/build.mjs'])
run(process.execPath, ['--test', 'tests/build.test.mjs'])
run('npm', ['pack', '--dry-run', '--json'], { shell: process.platform === 'win32' })
console.log('check complete')
