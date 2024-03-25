import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import { Button } from '@mantine/core'
import { IconSettingsCog } from '@tabler/icons-react'
import { QuestionTable } from './components/QuestionTable'
import { v4 as uuidv4 } from 'uuid'
import { Slider } from '@mantine/core'

import './App.css'
import { API_URL } from './env.ts'

export interface Question {
  id: string
  text: string
  score: number
}

const testQuestions = [
  { id: '2', text: 'What is the capital of Spain?', score: 0.8 },
  { id: '4', text: 'What is the capital of Germany?', score: 0.6 },
  { id: '1', text: 'What is the capital of France?', score: 0.9 },
  { id: '3', text: 'What is the capital of Italy?', score: 0.7 },
  { id: '8', text: 'What is the capital of South Korea?', score: 0.2 },
  { id: '5', text: 'What is the capital of Russia?', score: 0.5 },
  { id: '7', text: 'What is the capital of Japan?', score: 0.3 },
  { id: '9', text: 'What is the capital of North Korea?', score: 0.1 },
  { id: '6', text: 'What is the capital of China?', score: 0.4 },
]

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [selection, setSelection] = useState<string[]>([])
  const [questions, setQuestions] = useState<Question[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [similarityThreshold, setSimilarityThreshold] = useState(0)

  const generateQuestions = async () => {
    setLoading(true)
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch(`${API_URL}/generate_questions`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        const questionsWithId = data.questions.map(
          (question: { text: string; similarity_score: number }) => {
            return {
              id: uuidv4(),
              text: question.text,
              score: question.similarity_score,
            }
          }
        )
        setQuestions(questionsWithId)
        // JUST TO TEST SLIDER
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const genTestQuestions = () => {
    setQuestions(testQuestions)
  }

  return (
    <MantineProvider theme={theme}>
      <div className="body">
        <div className="upload-section">
          <DragDrop
            files={files}
            setFiles={setFiles}
          />

          <FileList
            files={files}
            setFiles={setFiles}
          />
        </div>

        <div className="generate-section">
          <Button
            loading={loading}
            onClick={genTestQuestions}
            disabled={files.length === 0 || loading}
            className="generate-question-btn"
          >
            <IconSettingsCog /> Generate Questions
          </Button>
        </div>

        <div className="config-section">
          <div id="title">Similarity Threshold</div>
          <Slider
            value={similarityThreshold}
            onChange={setSimilarityThreshold}
            color="blue"
            size="xl"
            min={0}
            max={1}
            marks={[
              { value: 0.25, label: '0.25' },
              { value: 0.5, label: '0.50' },
              { value: 0.75, label: '0.75' },
            ]}
            step={0.01}
            className="similarity-slider"
          />
        </div>

        <div className="question-section">
          <QuestionTable
            selection={selection}
            setSelection={setSelection}
            questions={questions}
            similarityThreshold={similarityThreshold}
          />
        </div>
      </div>
    </MantineProvider>
  )
}
