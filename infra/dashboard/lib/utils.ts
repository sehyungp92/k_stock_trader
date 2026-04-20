import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatKRW(value: number): string {
  const abs = Math.abs(value)
  const sign = value < 0 ? '-' : ''

  if (abs >= 1_000_000_000) {
    return `${sign}KRW ${(abs / 1_000_000_000).toFixed(1)}B`
  }
  if (abs >= 1_000_000) {
    return `${sign}KRW ${(abs / 1_000_000).toFixed(1)}M`
  }
  if (abs >= 1_000) {
    return `${sign}KRW ${(abs / 1_000).toFixed(1)}K`
  }
  return `${sign}KRW ${abs.toLocaleString()}`
}
