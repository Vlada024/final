import React from 'react'
import { createRoot } from 'react-dom/client'
import DotCounterTrainer from './components/DotCounterTrainer'
import './styles.css'

const App = () => {
  return <DotCounterTrainer />
}

const container = document.getElementById('root')
const root = createRoot(container)
root.render(<App />)
