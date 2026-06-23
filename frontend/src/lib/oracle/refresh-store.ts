import { create } from 'zustand'

type RefreshState = {
  autoRefresh: boolean
  setAutoRefresh: (v: boolean) => void
}

// Global toggle for the 60s auto-refresh. Hooks read this to decide their
// refetchInterval; the header exposes the switch + a manual Refresh button.
export const useRefreshStore = create<RefreshState>((set) => ({
  autoRefresh: true,
  setAutoRefresh: (v) => set({ autoRefresh: v }),
}))

export const REFRESH_MS = 60_000
